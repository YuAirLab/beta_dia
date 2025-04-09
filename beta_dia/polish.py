import numpy as np
import pandas as pd
from numba import jit

from beta_dia.log import Logger
from beta_dia import param_g
from beta_dia import utils

logger = Logger.get_logger()

try:
    # profile
    profile = lambda x: x
except:
    profile = lambda x: x

@jit(nopython=True, nogil=True)
def is_fg_share(fg_mz_1, fg_mz_2, tol_ppm):
    x, y = fg_mz_1.reshape(-1, 1), fg_mz_2.reshape(1, -1)

    delta_mz = np.abs(x - y)
    ppm = delta_mz / (x + 1e-7) * 1e6
    ppm_b = ppm < tol_ppm
    is_share_x = np.array([ppm_b[i, :].any() for i in range(len(ppm_b))])
    is_share_x = is_share_x & (fg_mz_1 > 0)

    return is_share_x


@jit(nopython=True, nogil=True, parallel=False)
def polish_prs_core(swath_id_v, measure_locus_v, measure_im_v,
                        fg_mz_m, tol_locus, tol_im, tol_ppm, sa_m):
    # The big fish eats the small fish.
    # If a fg ion shared by more confident pr, the sa and fg_mz will be zeros.
    for i in range(len(swath_id_v)):
        swath_id_i = swath_id_v[i]
        measure_locus_i = measure_locus_v[i]
        measure_im_i = measure_im_v[i]
        fg_mz_i = fg_mz_m[i]

        for j in range(i + 1, len(swath_id_v)):
            swath_id_j = swath_id_v[j]
            if swath_id_i != swath_id_j:
                break

            measure_locus_j = measure_locus_v[j]
            if abs(measure_locus_i - measure_locus_j) > tol_locus:
                continue

            measure_im_j = measure_im_v[j]
            if abs(measure_im_i - measure_im_j) > tol_im:
                continue

            sa_i = sa_m[i]
            fg_mz_j = fg_mz_m[j]

            is_share_v = is_fg_share(fg_mz_j, fg_mz_i, tol_ppm)
            is_share_v = is_share_v & (sa_i > 0)
            for jj in np.where(is_share_v)[0]:
                sa_m[j, jj] = 0
                fg_mz_m[j, jj] = 0

    return sa_m, fg_mz_m


def polish_prs(df_input, tol_im=0.03, tol_ppm=20, tol_sa_ratio=0.75, tol_share_num=5):
    '''
    1. Co-fragmentation prs should be polished.
    2. Decoy prs with cscore less than min(target) should be removed (option).
    '''
    df_input_target = df_input[df_input['decoy'] == 0].reset_index(drop=True)
    df_input_decoy = df_input[df_input['decoy'] == 1].reset_index(drop=True)

    df_v = []
    for df_target in [df_input_target, df_input_decoy]:
        target_num_before = len(df_target)

        # process I/L peptideform
        df_target['pr_IL'] = df_target['pr_id'].replace(['I', 'L'], ['x', 'x'], regex=True)
        idx_max = df_target.groupby('pr_IL')['cscore_pr_run'].idxmax()
        polish_IL_num = len(df_target) - len(idx_max)
        df_target = df_target.loc[idx_max].reset_index(drop=True)
        del df_target['pr_IL']

        # tol_locus is from the half of span
        spans = df_target.loc[df_target['q_pr_run'] < 0.01, 'score_elute_span']
        tol_locus = np.ceil(0.5 * spans.median())

        df_target = df_target.sort_values(
            by=['swath_id', 'cscore_pr_run'],
            ascending=[True, False],
            ignore_index=True
        )

        swath_id_v = df_target['swath_id'].values
        measure_locus_v = df_target['locus'].values
        measure_im_v = df_target['measure_im'].values

        cols_center = ['fg_mz_' + str(i) for i in range(param_g.fg_num)]
        fg_mz_center = df_target[cols_center].values
        cols_center = ['score_ion_sa_' + str(i) for i in range(2, 14)]
        sa_center = df_target[cols_center].values

        fg_mz_m = np.concatenate([fg_mz_center], axis=1)
        fg_mz_m = np.ascontiguousarray(fg_mz_m)
        fg_mz_m_raw = fg_mz_m.copy()

        sa_m = np.concatenate([sa_center], axis=1)
        sa_m = np.ascontiguousarray(sa_m)
        sa_m_raw = sa_m.copy()

        sa_m, fg_mz_m = polish_prs_core(
            swath_id_v, measure_locus_v, measure_im_v,
            fg_mz_m, tol_locus, tol_im, tol_ppm, sa_m
        )

        # screen
        nonshare_ratio = sa_m.sum(axis=1) / (1e-6 + sa_m_raw.sum(axis=1))
        good_condition1 = nonshare_ratio > tol_sa_ratio
        good_condition2 = (fg_mz_m_raw > fg_mz_m).sum(axis=1) < tol_share_num
        good_idx = good_condition1 & good_condition2
        df_target = df_target.iloc[good_idx]

        polish_bad_num = len(good_idx) - sum(good_idx)
        info = 'Removing dubious prs: {}-{}-{}={}'.format(
            target_num_before, polish_IL_num, polish_bad_num, len(df_target)
        )
        logger.info(info)
        df_v.append(df_target)

    df = pd.concat(df_v, ignore_index=True)
    if param_g.is_compare_mode:
        utils.cal_acc_recall(param_g.ws_single, df[df['decoy'] == 0], diann_q_pr=0.01)
    return df


def polish_prs_in_reanalysis(df_input, tol_im, tol_ppm, tol_sa_ratio, tol_share_num):
    '''
    1. Co-fragmentation prs should be polished.
    2. Decoy prs with cscore less than min(target) should be removed.
    '''
    bad_seqs = []
    target_good_idx = (df_input['decoy'] == 0) & (df_input['is_main'])

    df_target = df_input[target_good_idx].reset_index(drop=True)
    df_other = df_input[~target_good_idx]
    target_num_before = len(df_target)

    # process I/L peptideform
    df_target['pr_IL'] = df_target['pr_id'].replace(['I', 'L'], ['x', 'x'], regex=True)
    idx_max = df_target.groupby('pr_IL')['cscore_pr_run'].idxmax()
    polish_IL_num = len(df_target) - len(idx_max)
    df_target = df_target.loc[idx_max].reset_index(drop=True)
    del df_target['pr_IL']

    # tol_locus is from the half of span
    spans = df_target.loc[df_target['q_pr_run'] < 0.01, 'score_elute_span']
    tol_locus = np.ceil(0.5 * spans.median())

    df_target = df_target.sort_values(
        by=['swath_id', 'cscore_pr_run'],
        ascending=[True, False],
        ignore_index=True
    )

    swath_id_v = df_target['swath_id'].values
    measure_locus_v = df_target['locus'].values
    measure_im_v = df_target['measure_im'].values

    cols_center = ['fg_mz_' + str(i) for i in range(param_g.fg_num)]
    fg_mz_center = df_target[cols_center].values
    cols_center = ['score_ion_sa_' + str(i) for i in range(2, 14)]
    sa_center = df_target[cols_center].values

    fg_mz_m = np.concatenate([fg_mz_center], axis=1)
    fg_mz_m = np.ascontiguousarray(fg_mz_m)
    fg_mz_m_raw = fg_mz_m.copy()
    sa_m = np.concatenate([sa_center], axis=1)
    sa_m = np.ascontiguousarray(sa_m)
    sa_m_raw = sa_m.copy()

    sa_m, fg_mz_m = polish_prs_core(
        swath_id_v, measure_locus_v, measure_im_v,
        fg_mz_m, tol_locus, tol_im, tol_ppm, sa_m
    )

    # screen
    nonshare_ratio = sa_m.sum(axis=1) / (1e-6 + sa_m_raw.sum(axis=1))
    good_condition1 = nonshare_ratio > tol_sa_ratio
    good_condition2 = (fg_mz_m_raw > fg_mz_m).sum(axis=1) < tol_share_num
    good_idx = good_condition1 & good_condition2
    df_target.loc[~good_idx, 'is_main'] = False
    bad_seqs.extend(df_target.loc[~good_idx, 'pr_id'].tolist())

    polish_bad_num = len(good_idx) - sum(good_idx)
    info = 'Removing dubious target prs: {}-{}-{}={}'.format(
        target_num_before, polish_IL_num, polish_bad_num, df_target['is_main'].sum()
    )
    logger.info(info)

    df = pd.concat([df_target, df_other], ignore_index=True)

    if param_g.is_compare_mode:
        utils.cal_acc_recall(param_g.ws_single, df[df['decoy'] == 0], diann_q_pr=0.01)

    return df, bad_seqs


@jit(nopython=True, nogil=True, parallel=False)
def zero_interference_core(swath_id_v, measure_locus_v, measure_im_v,
                        fg_mz_m, area_m, sa_m, tol_locus, tol_im, tol_ppm,
                           other_idx):
    # The big fish eats the small fish.
    # If a fg ion shared by more confident pr, the sa and fg_mz will be zeros.
    for i in range(other_idx):
        swath_id_i = swath_id_v[i]
        measure_locus_i = measure_locus_v[i]
        measure_im_i = measure_im_v[i]
        fg_mz_i = fg_mz_m[i]

        for j in range(other_idx, len(swath_id_v)):
            swath_id_j = swath_id_v[j]
            if swath_id_i != swath_id_j:
                continue

            measure_locus_j = measure_locus_v[j]
            if abs(measure_locus_i - measure_locus_j) > tol_locus:
                continue

            measure_im_j = measure_im_v[j]
            if abs(measure_im_i - measure_im_j) > tol_im:
                continue

            fg_mz_j = fg_mz_m[j]

            is_share_v = is_fg_share(fg_mz_j, fg_mz_i, tol_ppm)
            for jj in np.where(is_share_v)[0]:
                sa_m[j, jj] = 0
                area_m[j, jj] = 0

    return area_m, sa_m


def zero_interference_areas(df_input, tol_locus=3, tol_im=0.03, tol_ppm=20):
    # This function is calculated for a run
    # only main prs eat other prs
    df_target = df_input[df_input['decoy'] == 0].copy()
    df_decoy = df_input[df_input['decoy'] == 1].copy()

    df_target = df_target.sort_values(
        by=['cscore_pr_run'], # 'cscore_pr_global
        ascending=[False],
        ignore_index=True
    ) # NA is positing end

    swath_id_v = df_target['swath_id'].values
    measure_locus_v = df_target['locus'].values
    measure_im_v = df_target['measure_im'].values

    cols_center = ['fg_mz_' + str(i) for i in range(param_g.fg_num)]
    fg_mz_m = df_target[cols_center].values
    fg_mz_m = np.ascontiguousarray(fg_mz_m)

    cols_quant = ['score_ion_quant_' + str(i) for i in range(2, 14)]
    area_m = df_target[cols_quant].values
    area_m = np.ascontiguousarray(area_m)

    cols_sa = ['score_ion_sa_' + str(i) for i in range(2, 14)]
    sa_m = df_target[cols_sa].values
    sa_m = np.ascontiguousarray(sa_m)

    other_idx = np.where(df_target['cscore_pr_run'].isna())[0][0]

    area_m, sa_m = zero_interference_core(
        swath_id_v, measure_locus_v, measure_im_v,
        fg_mz_m, area_m, sa_m, tol_locus, tol_im, tol_ppm, other_idx
    )
    bad_idx = (area_m == 0) | (sa_m == 0)
    area_m[bad_idx] = 0.
    sa_m[bad_idx] = 0.
    df_target[cols_quant] = area_m
    df_target[cols_sa] = sa_m

    df = pd.concat([df_target, df_decoy], ignore_index=True)
    return df