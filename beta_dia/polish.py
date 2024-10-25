import numpy as np
import pandas as pd
from numba import jit

from beta_dia.log import Logger
from beta_dia import param_g

logger = Logger.get_logger()

@jit(nopython=True, nogil=True)
def is_fg_share(fg_mz_1, fg_mz_2, tol_ppm):
    x, y = fg_mz_1.reshape(-1, 1), fg_mz_2.reshape(1, -1)

    delta_mz = np.abs(x - y)
    ppm = delta_mz / (x + 1e-7) * 1e6
    ppm_b = ppm < tol_ppm
    is_share_x = np.array([ppm_b[i, :].any() for i in range(len(ppm_b))])
    is_share_x = is_share_x & (fg_mz_1 > 0)

    return is_share_x


def polish_prs_in5(df, tol_im, tol_ppm, tol_sa_ratio):
    '''
    1. Co-fragmentation prs should be polished.
    2. Decoy prs with cscore less than min(target) should be removed.
    '''
    # logger.info('Removing dubious target prs in 5%-FDR...')

    assert df['group_rank'].max() == 1

    target_good_idx = (df['decoy'] == 0) & (df['q_pr'] < 0.05)
    df_target = df[target_good_idx]
    df_other = df[~target_good_idx]
    target_num_before = len(df_target)

    # tol_locus is from the half of span
    spans = df_target.loc[df_target['q_pr'] < 0.01, 'score_elute_span']
    tol_locus = np.ceil(0.5 * spans.median())

    df_target = df_target.sort_values(
        by=['swath_id'],
        ascending=[True],
        ignore_index=True
    )

    swath_id_v = df_target['swath_id'].values
    measure_locus_v = df_target['locus'].values
    measure_im_v = df_target['measure_im'].values

    cols_center = ['fg_mz_' + str(i) for i in range(param_g.fg_num)]
    fg_mz_center = df_target[cols_center].values
    cols_center = ['score_center_elution_' + str(i) for i in range(2, 14)]
    sa_center = df_target[cols_center].values

    cols_1H = ['fg_mz_1H_' + str(i) for i in range(param_g.fg_num)]
    fg_mz_1H = df_target[cols_1H].values
    cols_1H = ['score_1H_elution_' + str(i) for i in range(2, 14)]
    sa_1H = df_target[cols_1H].values

    pr_mz_unfrag = df_target['pr_mz'].values.reshape(-1, 1)
    sa_pr_unfrag = df_target['score_center_elution_1'].values.reshape(-1, 1)

    fg_mz_m = np.concatenate([pr_mz_unfrag, fg_mz_center, fg_mz_1H], axis=1)
    fg_mz_m = np.ascontiguousarray(fg_mz_m)
    sa_m = np.concatenate([sa_pr_unfrag, sa_center, sa_1H], axis=1)
    sa_m = np.ascontiguousarray(sa_m)

    is_dubious_m, competer_pr_num = polish_prs_in5_core(
        swath_id_v, measure_locus_v, measure_im_v,
        fg_mz_m, tol_locus, tol_im, tol_ppm
    )
    disturb_ratios = (is_dubious_m * sa_m).sum(axis=1) / (sa_m.sum(axis=1) + 1e-7)
    df_target = df_target[disturb_ratios < tol_sa_ratio].reset_index(drop=True)

    # remove one-by-one
    # while True:
    #     is_dubious_m, competer_pr_num = polish_prs_core1(
    #         swath_id_v, measure_locus_v, measure_im_v,
    #         fg_mz_m,
    #         tol_locus, tol_im, tol_ppm
    #     )
    #     disturb_ratios = (is_dubious_m * sa_m).sum(axis=1) / (sa_m.sum(axis=1) + 1e-7)
    #     df_target['competer_num'] = competer_pr_num
    #     df_target['disturb_ratio'] = disturb_ratios
    #
    #     if disturb_ratios.max() > tol_sa_ratio:
    #         # 删除disturb_ratios最大值的行，如果有多个，先删除cscore_pr最小的行
    #         max_rows = disturb_ratios == disturb_ratios.max()
    #         dfx = df_target[max_rows]
    #         dfx = dfx.sort_values(by=['competer_num', 'cscore_pr'], ascending=[False, True])
    #         drop_idx = dfx.index[0]
    #         idx = df_target.index.values != drop_idx
    #         df_target = df_target.drop(drop_idx).reset_index(drop=True)
    #         swath_id_v = swath_id_v[idx]
    #         measure_locus_v = measure_locus_v[idx]
    #         measure_im_v = measure_im_v[idx]
    #         fg_mz_m = fg_mz_m[idx]
    #         sa_m = sa_m[idx]
    #     else:
    #         break

    # result
    target_num_now = len(df_target)
    df = pd.concat([df_target, df_other], ignore_index=True)
    info = 'Removing dubious target prs in 5%-FDR: {}/{}'.format(
        target_num_before - target_num_now, target_num_before
    )
    logger.info(info)

    return df


@jit(nopython=True, nogil=True, parallel=False)
def polish_prs_in5_core(swath_id_v, measure_locus_v, measure_im_v,
                        fg_mz_m, tol_locus, tol_im, tol_ppm):
    is_dubious_m = np.zeros_like(fg_mz_m, dtype=np.bool_)
    competor_pr_nums = np.zeros_like(swath_id_v)

    for i in range(len(swath_id_v)):
        swath_id_i = swath_id_v[i]
        measure_locus_i = measure_locus_v[i]
        measure_im_i = measure_im_v[i]
        fg_mz_i = fg_mz_m[i]

        for j in range(len(swath_id_v)):
            if j == i:
                continue

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
            is_share_v = is_fg_share(fg_mz_i, fg_mz_j, tol_ppm)
            for k in range(len(is_share_v)):
                is_dubious_m[i, k] |= is_share_v[k]

            if np.sum(is_share_v) > 0:
                competor_pr_nums[i] += 1

    return is_dubious_m, competor_pr_nums


def polish_prs_over5(df, tol_im, tol_ppm, tol_sa_ratio):
    '''
    1. Co-fragmentation prs should be polished.
    2. Decoy prs with cscore less than min(target) should be removed.
    '''
    # logger.info('Removing dubious target prs over 5%-FDR...')

    assert df['group_rank'].max() == 1

    # tol_locus is from the half of span
    spans = df.loc[df['q_pr'] < 0.01, 'score_elute_span']
    tol_locus = np.ceil(0.5 * spans.median())
    # logger.info(f'Span median is: {spans.median()}, tol_locus is: {tol_locus}')

    df = df.sort_values(
        by=['swath_id', 'cscore_pr'],
        ascending=[True, False],
        ignore_index=True
    )

    swath_id_v = df['swath_id'].values
    measure_locus_v = df['locus'].values
    measure_im_v = df['measure_im'].values

    cols_center = ['fg_mz_' + str(i) for i in range(param_g.fg_num)]
    fg_mz_center = df[cols_center].values
    cols_center = ['score_center_elution_' + str(i) for i in range(2, 14)]
    sa_center = df[cols_center].values

    cols_1H = ['fg_mz_1H_' + str(i) for i in range(param_g.fg_num)]
    fg_mz_1H = df[cols_1H].values
    cols_1H = ['score_1H_elution_' + str(i) for i in range(2, 14)]
    sa_1H = df[cols_1H].values

    pr_mz_unfrag = df['pr_mz'].values.reshape(-1, 1)
    sa_pr_unfrag = df['score_center_elution_1'].values.reshape(-1, 1)

    fg_mz_m = np.concatenate([pr_mz_unfrag, fg_mz_center, fg_mz_1H], axis=1)
    fg_mz_m = np.ascontiguousarray(fg_mz_m)
    sa_m = np.concatenate([sa_pr_unfrag, sa_center, sa_1H], axis=1)
    sa_m = np.ascontiguousarray(sa_m)

    q_value_v = df['q_pr'].values

    is_dubious_m = polish_prs_over5_core(
            swath_id_v, measure_locus_v, measure_im_v,
            fg_mz_m, tol_locus, tol_im, tol_ppm, q_value_v
        )
    disturb_ratios = (is_dubious_m * sa_m).sum(axis=1) / (sa_m.sum(axis=1) + 1e-7)
    target_num_before = len(df)
    is_dubious_v = disturb_ratios >= tol_sa_ratio
    df['is_dubious'] = is_dubious_v
    df = df[~is_dubious_v].reset_index(drop=True)

    # result
    remove_num = target_num_before - len(df)
    info = 'Removing dubious target prs over 5%-FDR: {}/{}'.format(
        remove_num, target_num_before
    )
    logger.info(info)

    return df


@jit(nopython=True, nogil=True, parallel=False)
def polish_prs_over5_core(swath_id_v, measure_locus_v, measure_im_v,
                          fg_mz_m, tol_locus, tol_im, tol_ppm, q_value_v):
    '''
    Each thread processes a pr with cscore_pr ascending.
    '''
    is_dubious_m = np.zeros_like(fg_mz_m, dtype=np.bool_)

    for i in range(len(swath_id_v)):
        swath_id_i = swath_id_v[i]
        measure_locus_i = measure_locus_v[i]
        measure_im_i = measure_im_v[i]
        fg_mz_i = fg_mz_m[i]

        for j in range(i+1, len(swath_id_v)):
            q_value_j = q_value_v[j]
            if q_value_j < 0.05:
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
            is_share_v = is_fg_share(fg_mz_j, fg_mz_i, tol_ppm)
            for k in range(len(is_share_v)):
                is_dubious_m[j, k] |= is_share_v[k]

    return is_dubious_m
