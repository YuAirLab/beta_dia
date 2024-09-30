import numpy as np
import pandas as pd
from numba import jit

from beta_dia.log import Logger
from beta_dia import param_g

logger = Logger.get_logger()

@jit(nopython=True, nogil=True)
def cal_fg_share_num(fg_mz_1, fg_mz_2, tol_ppm):
    fg_mz_1 = fg_mz_1.reshape(-1, 1)
    fg_mz_2 = fg_mz_2.reshape(1, -1)
    delta_mz = np.abs(fg_mz_1 - fg_mz_2)
    ppm_delta_mz = delta_mz / (fg_mz_1 + 1e-7) * 1e6
    share_num = np.sum((ppm_delta_mz < tol_ppm) & (fg_mz_1 > 0))
    return share_num


@jit(nopython=True, nogil=True)
def is_fg_share(fg_mz_1, fg_mz_2, tol_ppm):
    x, y = fg_mz_1.reshape(-1, 1), fg_mz_2.reshape(1, -1)

    delta_mz = np.abs(x - y)
    ppm = delta_mz / (x + 1e-7) * 1e6
    ppm_b = ppm < tol_ppm
    is_share_x = np.array([ppm_b[i, :].any() for i in range(len(ppm_b))])
    is_share_x = is_share_x & (fg_mz_1 > 0)

    is_share_y = np.array([ppm_b[:, i].any() for i in range(len(ppm_b))])
    is_share_y = is_share_y & (fg_mz_2 > 0)

    return is_share_x, is_share_y

@jit(nopython=True, nogil=True, parallel=False)
def polish_core(swath_id_v, measure_locus_v, measure_im_v,
                fg_mz_m, is_share_m, cscore_v,
                tol_locus, tol_im, tol_ppm, related_pr_num_v, is_related_best_v):
    '''
    Each thread processes a pr with cscore_pr ascending.
    '''
    for i in range(len(swath_id_v)):
        swath_id_i = swath_id_v[i]
        measure_locus_i = measure_locus_v[i]
        measure_im_i = measure_im_v[i]
        fg_mz_i = fg_mz_m[i]
        cscore_i = cscore_v[i]

        for j in range(i+1, len(swath_id_v)):

            swath_id_j = swath_id_v[j]
            if swath_id_i != swath_id_j:
                break

            measure_locus_j = measure_locus_v[j]
            if abs(measure_locus_i - measure_locus_j) > tol_locus:
                break

            measure_im_j = measure_im_v[j]
            if abs(measure_im_i - measure_im_j) > tol_im:
                break

            fg_mz_j = fg_mz_m[j]
            share_num = cal_fg_share_num(fg_mz_i, fg_mz_j, tol_ppm)
            if share_num <= 0:
                continue

            is_share_x, is_share_y = is_fg_share(fg_mz_i, fg_mz_j, tol_ppm)
            is_share_m[i] |= is_share_x
            is_share_m[j] |= is_share_y

            related_pr_num_v[i] += 1
            related_pr_num_v[j] += 1

            cscore_j = cscore_v[j]
            if cscore_i > cscore_j:
                is_related_best_v[i] = True
            else:
                is_related_best_v[j] = True


def polish_after_qvalue(df, tol_locus, tol_im, tol_ppm, tol_share_num):
    '''
    Polish target prs after the calculation of qvalues.
    Args:
        df: already q cut at 5% FDR

    Returns:
        df with ['share_num'] to refer a better pr leading this id.
    '''
    pr_num_before = len(df[(df['q_pr'] < 0.01) & (df['decoy'] == 0)])

    assert df['q_pr'].max() <= 0.05
    assert df['group_rank'].max() == 1

    df_target = df[df['decoy'] == 0]
    df_decoy = df[df['decoy'] == 1]

    df_target = df_target.sort_values(
        by=['swath_id', 'locus', 'measure_im'],
        ascending=[True, True, True],
        ignore_index=True
    )

    swath_id_v = df_target['swath_id'].values
    measure_locus_v = df_target['locus'].values
    measure_im_v = df_target['measure_im'].values
    cols_center = ['fg_mz_' + str(i) for i in range(param_g.fg_num)]
    fg_mz_m = df_target[cols_center].values
    cscore_v = df_target['cscore_pr'].values

    is_fg_share_m = np.zeros((len(swath_id_v), fg_mz_m.shape[-1]), dtype=bool)
    related_pr_num_v = np.zeros_like(swath_id_v)
    is_related_best_v = np.zeros_like(swath_id_v, dtype=bool)

    polish_core(
        swath_id_v, measure_locus_v, measure_im_v,
        fg_mz_m, is_fg_share_m, cscore_v,
        tol_locus, tol_im, tol_ppm, related_pr_num_v, is_related_best_v
    )

    fg_share_num_v = is_fg_share_m.sum(axis=1)
    assert fg_share_num_v.max() <= is_fg_share_m.shape[-1]
    assert sum((fg_share_num_v >= 1) & (related_pr_num_v <= 0)) == 0

    df_target['share_fg_num'] = fg_share_num_v
    df_target['related_pr_num'] = related_pr_num_v
    df_target['related_best'] = is_related_best_v

    # df_target.loc[
    #     (df_target['share_fg_num'] >= tol_share_num) &
    #     (df_target['related_pr_num'] >= 2), 'q_pr'] = 1
    # df_target.loc[
    #     (df_target['share_fg_num'] >= tol_share_num) &
    #     (df_target['related_best'] == False), 'q_pr'] = 1

    df_target.loc[df_target['share_fg_num'] >= tol_share_num, 'q_pr'] = 1

    df_decoy['share_fg_num'] = -1
    df_decoy['related_pr_num'] = -1
    df_decoy['related_best'] = False
    df = pd.concat([df_target, df_decoy], ignore_index=True)

    pr_num_after = len(df[(df['q_pr'] < 0.01) & (df['decoy'] == 0)])
    info = 'Prs with share fgs before polish: {}, after: {}'.format(
        pr_num_before, pr_num_after
    )
    logger.info(info)

    return df



@jit(nopython=True, nogil=True, parallel=False)
def polish_prs_core(swath_id_v, measure_locus_v, measure_im_v,
                fg_mz_m,
                tol_locus, tol_im, tol_ppm, tol_share_num, is_dubious_v):
    '''
    Each thread processes a pr with cscore_pr ascending.
    '''
    for i in range(len(swath_id_v)):
        swath_id_i = swath_id_v[i]
        measure_locus_i = measure_locus_v[i]
        measure_im_i = measure_im_v[i]
        fg_mz_i = fg_mz_m[i]

        for j in range(i+1, len(swath_id_v)):
            is_dubious = is_dubious_v[j]
            if is_dubious:
                continue

            swath_id_j = swath_id_v[j]
            if swath_id_i != swath_id_j:
                break

            measure_locus_j = measure_locus_v[j]
            if abs(measure_locus_i - measure_locus_j) > tol_locus:
                continue

            measure_im_j = measure_im_v[j]
            if abs(measure_im_i - measure_im_j) > tol_im:
                continue

            fg_mz_j = fg_mz_m[j]
            share_num = cal_fg_share_num(fg_mz_i, fg_mz_j, tol_ppm)
            if share_num >= tol_share_num:
                is_dubious_v[j] = True


def polish_prs(df, tol_im, tol_ppm, tol_share_num):
    '''
    1. Co-fragmentation prs should be polished.
    2. Decoy prs with cscore less than min(target) should be removed.
    '''
    logger.info('Polishing dubious target prs and low confidence decoy prs...')

    assert df['group_rank'].max() == 1

    df_target = df[df['decoy'] == 0]
    df_decoy = df[df['decoy'] == 1]

    # tol_locus is from the half of span
    spans = df_target.loc[df_target['q_pr'] < 0.01, 'score_elute_span']
    tol_locus = np.ceil(0.5 * spans.median())
    logger.info(f'Span median is: {spans.median()}, tol_locus is: {tol_locus}')

    df_target = df_target.sort_values(
        by=['swath_id', 'cscore_pr'],
        ascending=[True, False],
        ignore_index=True
    )

    swath_id_v = df_target['swath_id'].values
    measure_locus_v = df_target['locus'].values
    measure_im_v = df_target['measure_im'].values
    cols_center = ['fg_mz_' + str(i) for i in range(param_g.fg_num)]
    fg_mz_m = np.ascontiguousarray(df_target[cols_center].values)

    is_dubious_v = np.zeros_like(swath_id_v, dtype=bool)

    polish_prs_core(
        swath_id_v, measure_locus_v, measure_im_v,
        fg_mz_m,
        tol_locus, tol_im, tol_ppm, tol_share_num, is_dubious_v
    )

    target_num_before = len(df_target)
    df_target = df_target[~is_dubious_v]

    # removing low confidence decoy prs
    cut = df_target['cscore_pr'].min()
    bad_decoys_num = (df_decoy['cscore_pr'] < cut).sum()
    decoy_num_before = len(df_decoy)
    df_decoy = df_decoy[df_decoy['cscore_pr'] > cut]

    # result
    df = pd.concat([df_target, df_decoy], ignore_index=True)
    info = 'Remove dubious target prs: {}/{}, bad decoys prs: {}/{}'.format(
        sum(is_dubious_v), target_num_before, bad_decoys_num, decoy_num_before
    )
    logger.info(info)

    return df