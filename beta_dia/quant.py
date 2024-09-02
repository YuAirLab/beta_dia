from itertools import product

import numpy as np
import pandas as pd
from numba import cuda, jit, prange

from beta_dia import fxic
from beta_dia import param_g
from beta_dia import utils
from beta_dia.log import Logger

logger = Logger.get_logger()

def cross_cos(x):
    norms = np.linalg.norm(x, axis=1) + 1e-6
    normalized_x = x / norms[:, np.newaxis]
    cosine_sim = np.dot(normalized_x, normalized_x.T)
    return cosine_sim


@jit(nopython=True, nogil=True, parallel=True)
def interp_xics(xics, rts, target_dim):
    '''
    xics: [n_pep, n_ion, n_cycle]
    rts: [n_pep, n_cycle]
    result_xics: [n_pep, n_ion, target_dim]
    result_rts: [n_pep, target_dim]
    '''
    n_pep, n_ion, n_cycle = xics.shape
    result_xics = np.zeros((n_pep, n_ion, target_dim))
    result_rts = np.zeros((n_pep, target_dim))
    for i_pep in prange(n_pep):
        rts_pep = rts[i_pep]
        rts_interp = np.linspace(rts_pep[0], rts_pep[-1], target_dim)
        result_rts[i_pep] = rts_interp
        for i_ion in range(n_ion):
            xic_raw = xics[i_pep, i_ion]
            xic_interp = np.interp(rts_interp, rts_pep, xic_raw)
            result_xics[i_pep, i_ion] = xic_interp
    return result_rts, result_xics


def grid_xic_best(df_batch, ms1_centroid, ms2_centroid):
    locus_start_v = df_batch['score_elute_span_left'].values
    locus_end_v = df_batch['score_elute_span_right'].values

    tol_ppm_v = [20., 16., 12., 8., 4.]
    tol_im_v = [0.02, 0.01]
    grid_params = list(product(tol_ppm_v, tol_im_v))

    xics_v = []
    expand_dim = 64
    for search_i, (tol_ppm, tol_im) in enumerate(grid_params):
        _, rts, _, _, xics = fxic.extract_xics(
            df_batch,
            ms1_centroid,
            ms2_centroid,
            im_tolerance=tol_im,
            ppm_tolerance=tol_ppm,
            cycle_num=13,
            by_pred=False
        )
        xics = xics.copy_to_host()[:, 2:8, :]
        mask1 = np.arange(xics.shape[2]) >= locus_start_v[:, None, None]
        mask2 = np.arange(xics.shape[2]) <= locus_end_v[:, None, None]
        xics = xics * mask1 * mask2
        rts, xics = interp_xics(xics, rts, expand_dim)
        xics = fxic.gpu_simple_smooth(cuda.to_device(xics))
        xics = xics.copy_to_host()

        # find best profile
        if search_i == 0:
            sas = np.array(list(map(cross_cos, xics)))
            sa_sum = sas.sum(axis=-1)
            best_ion_idx = sa_sum.argmax(axis=-1)
            best_profile = xics[np.arange(len(xics)), best_ion_idx]

            bad_xic = np.abs(best_profile.argmax(axis=-1) - expand_dim/2) > 6

            # boundary by best_profile
            box = best_profile > best_profile.max(axis=-1, keepdims=True) * 0.2
            left = box.argmax(axis=-1)
            right = expand_dim - 1 - box[:, ::-1].argmax(axis=-1)
            df_batch['integral_left'] = left
            df_batch['integral_right'] = right

        xics_v.append(xics)
    xics = np.transpose(np.stack(xics_v), (1, 0, 2, 3))

    # find other profile with the help of best_profile
    ion_num = xics.shape[2]
    best_profile = np.repeat(
        best_profile[:, None, :], len(grid_params), axis=1
    )
    best_profile = np.repeat(best_profile[:, :, None, :], ion_num, axis=2)
    dot_sum = (best_profile * xics).sum(axis=-1)
    norm1 = np.linalg.norm(best_profile, axis=-1) + 1e-6
    norm2 = np.linalg.norm(xics, axis=-1) + 1e-6
    sas = dot_sum / (norm1 * norm2)
    idx = sas.argmax(axis=1)

    idx_1 = idx.flatten()
    idx_0 = np.repeat(np.arange(len(xics)), ion_num)
    idx_2 = np.tile(np.arange(ion_num), len(xics))
    xics = xics[idx_0, idx_1, idx_2].reshape(len(xics), ion_num, -1)

    # interference correction
    best_profile = best_profile[:, 0, 0, :]
    r_m = xics / (best_profile[:, None, :] + 1e-7)
    r_center = r_m[:, :, int(expand_dim/2)]
    bad_idx = r_m > 1.5 * r_center[:, :, None]
    tmp = 1.5 * r_center[:, :, None] * best_profile[:, None, :]
    xics[bad_idx] = tmp[bad_idx]

    # bad_xic re-extract
    _, rts2, _, _, xics2 = fxic.extract_xics(
        df_batch,
        ms1_centroid,
        ms2_centroid,
        im_tolerance=0.025,
        ppm_tolerance=15,
        cycle_num=13,
        by_pred=False
    )
    xics2 = xics2.copy_to_host()[:, 2:8, :]
    mask1 = np.arange(xics2.shape[2]) >= locus_start_v[:, None, None]
    mask2 = np.arange(xics2.shape[2]) <= locus_end_v[:, None, None]
    xics2 = xics2 * mask1 * mask2
    rts2, xics2 = interp_xics(xics2, rts2, expand_dim)
    xics2 = fxic.gpu_simple_smooth(cuda.to_device(xics2))
    xics2 = xics2.copy_to_host()

    xics[bad_xic] = xics2[bad_xic]
    df_batch.loc[bad_xic, 'integral_left'] = 15 # 3-13, 15-64
    df_batch.loc[bad_xic, 'integral_right'] = 50 # 9-13, 50-64

    return rts, xics


def quant_pr(df, ms):
    # df['species'] = df['protein_name'].str.split('_').str[-1]
    # df = df[df['species'] == 'HUMAN'].reset_index(drop=True)
    # df = df[df['pr_id'] == 'M(UniMod:35)TDSFTEQADQVTAEVGK2']

    df_good = []
    for swath_id in df['swath_id'].unique():
        df_swath = df[df['swath_id'] == swath_id]
        df_swath = df_swath.reset_index(drop=True)

        # ms
        ms1_centroid, ms2_centroid = ms.copy_map_to_gpu(swath_id, centroid=True)

        # in batches
        batch_n = param_g.batch_xic_locus
        for batch_idx, df_batch in df_swath.groupby(df_swath.index // batch_n):
            df_batch = df_batch.reset_index(drop=True)

            # grid search for best profiles
            rts, xics = grid_xic_best(df_batch, ms1_centroid, ms2_centroid)

            # boundary
            locus_start_v = df_batch['integral_left'].values
            locus_end_v = df_batch['integral_right'].values
            mask1 = np.arange(xics.shape[2]) >= locus_start_v[:, None, None]
            mask2 = np.arange(xics.shape[2]) <= locus_end_v[:, None, None]
            xics = xics * mask1 * mask2

            # areas not using rts
            areas = np.trapz(xics, axis=-1)
            quant_pr = areas.sum(axis=1)
            df_batch['quant_pr'] = quant_pr.astype(np.float32)

            df_good.append(df_batch)

        utils.release_gpu_scans(ms1_centroid)
        utils.release_gpu_scans(ms2_centroid)

    df = pd.concat(df_good, axis=0, ignore_index=True)

    min_value =df.loc[df['quant_pr'] > 0, 'quant_pr'].min()
    df.loc[df['quant_pr'] <= 0., 'quant_pr'] = min_value
    assert df['quant_pr'].min() > 0
    return df


def quant_pg(df):
    '''
    If a pg has >=1 pr within 1%-FDR, mean value of top-3 if for its quant.
    Else, top-1 is its quant.
    '''
    df_tmp1 = df[df['q_pr'] < 0.01].reset_index(drop=True)
    df_tmp1 = df_tmp1.groupby('protein_group').apply(
        lambda x: x.nlargest(3, 'quant_pr')['quant_pr'].mean()
    ).reset_index(name='quant_pg')

    df_tmp2 = df[~df['protein_group'].isin(df_tmp1['protein_group'])]
    if len(df_tmp2) > 0:
        df_tmp2 = df_tmp2.reset_index(drop=True)
        df_tmp2 = df_tmp2.groupby('protein_group').apply(
            lambda x: x.nlargest(1, 'quant_pr')['quant_pr'].mean()
        ).reset_index(name='quant_pg')
        df_tmp = pd.concat([df_tmp1, df_tmp2])
    else:
        df_tmp = df_tmp1

    df = pd.merge(df, df_tmp, on='protein_group', how='outer')
    assert df['quant_pg'].min() > 0
    return df
