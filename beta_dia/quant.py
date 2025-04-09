from itertools import product

import numpy as np
import pandas as pd
from numba import cuda
import torch

from beta_dia import fxic
from beta_dia import param_g
from beta_dia import utils
from beta_dia.log import Logger

try:
    # profile
    profile = lambda x: x
except:
    profile = lambda x: x

logger = Logger.get_logger()

def mask_tensor(x, left, right):
    device = x.device

    n_pr, n_ion, n_cycle = x.shape
    cycle_indices = torch.arange(n_cycle, device=device).view(1, 1, -1)

    left = left.view(-1, 1, 1)
    right = right.view(-1, 1, 1)
    mask = (cycle_indices >= left) & (cycle_indices <= right)
    return x * mask


@profile
def interp_xics(x, rts_input, target_dim):
    rts = torch.from_numpy(rts_input).to(param_g.gpu_id)

    n_pr, n_ion, n_cycle = x.shape

    tmp = torch.linspace(0, 1, target_dim, device=x.device).unsqueeze(0)
    new_rts = rts[:, [0]] + (rts[:, [-1]] - rts[:, [0]]) * tmp

    idx = torch.searchsorted(rts, new_rts)
    idx_right = idx.clamp(max=n_cycle - 1)
    idx_left = (idx_right - 1).clamp(min=0)

    t_left = torch.gather(rts, 1, idx_left)
    t_right = torch.gather(rts, 1, idx_right)

    idx_left_exp = idx_left.unsqueeze(1).expand(n_pr, n_ion, target_dim)
    idx_right_exp = idx_right.unsqueeze(1).expand(n_pr, n_ion, target_dim)

    x_left = torch.gather(x, 2, idx_left_exp)
    x_right = torch.gather(x, 2, idx_right_exp)

    new_rts_exp = new_rts.unsqueeze(1)
    t_left_exp = t_left.unsqueeze(1)
    t_right_exp = t_right.unsqueeze(1)

    eps = 1e-8
    weight = (new_rts_exp - t_left_exp) / (t_right_exp - t_left_exp + eps)
    x_interp = x_left + weight * (x_right - x_left)

    return new_rts, x_interp


def select_other_profiles(x_profile, best_profile):
    device = x_profile.device
    n_pr, n_condition, n_ion, n_cycle = x_profile.shape

    best_profile_exp = best_profile.unsqueeze(1).unsqueeze(1)  # [n_pr, 1, 1, n_cycle]

    # dot
    dot = (x_profile * best_profile_exp).sum(dim=-1)

    # L2
    norm_x = x_profile.norm(dim=-1)  # [n_pr, n_condition, n_ion]
    norm_best = best_profile.norm(dim=-1).unsqueeze(1).unsqueeze(1) # [n_pr, 1, 1]

    # cos
    cos_sim = dot / (norm_x * norm_best + 1e-8)

    # best
    best_condition_idx = torch.argmax(cos_sim, dim=1)  # [n_pr, n_ion]

    # select xic
    pr_idx = torch.arange(n_pr, device=device).unsqueeze(1).expand(n_pr, n_ion)
    ion_idx = torch.arange(n_ion, device=device).unsqueeze(0).expand(n_pr, n_ion)
    best_x = x_profile[pr_idx, best_condition_idx, ion_idx, :] # [n_pr, n_ion, n_cycle]

    # select sa
    best_cos_sim = cos_sim[pr_idx, best_condition_idx, ion_idx]  # [n_pr, n_ion]
    best_cos_sim_clamped = best_cos_sim.clamp(-1, 1)
    angles = torch.acos(best_cos_sim_clamped)
    sas = 1 - 2 * angles / torch.pi
    assert sas.max().item() <= 1.
    assert sas.min().item() >= 0.

    return best_x, sas


def select_best_profile(x_profile):
    n_pr, n_ion, n_cycle = x_profile.shape

    x_norm = x_profile / (x_profile.norm(dim=-1, keepdim=True) + 1e-8)
    cos_sim = torch.matmul(x_norm, x_norm.transpose(1, 2))
    avg_sim = (cos_sim.sum(dim=-1) - 1.0) / (n_ion - 1)
    best_ion_idx = torch.argmax(avg_sim, dim=-1)

    pr_indices = torch.arange(n_pr, device=x_profile.device)
    best_profile = x_profile[pr_indices, best_ion_idx, :]

    return best_profile # [n_pr, n_cycle]


def interference_correction(xics, best_profile):
    r_m = xics / (best_profile[:, None, :] + 1e-7)
    r_center = r_m[:, :, int(xics.shape[-1] / 2)]
    bad_idx = r_m > 1.5 * r_center[:, :, None]
    tmp = 1.5 * r_center[:, :, None] * best_profile[:, None, :]
    xics[bad_idx] = tmp[bad_idx]
    return xics


@profile
def grid_xic_best(df_batch, ms1_centroid, ms2_centroid):
    locus_start_v = df_batch['score_elute_span_left'].values
    locus_start_v = torch.from_numpy(locus_start_v).to(param_g.gpu_id)

    locus_end_v = df_batch['score_elute_span_right'].values
    locus_end_v = torch.from_numpy(locus_end_v).to(param_g.gpu_id)

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
            by_pred=False,
        )

        xics = utils.convert_numba_to_tensor(xics) # 14 ions
        xics = mask_tensor(xics, locus_start_v, locus_end_v)
        rts, xics = interp_xics(xics, rts, expand_dim)
        # rts, xics = utils.interp_xics(xics, rts, expand_dim)
        xics = fxic.gpu_simple_smooth(cuda.as_cuda_array(xics))
        xics = utils.convert_numba_to_tensor(xics)

        # find best profile from top-6
        if search_i == 0:
            xics_top6 = xics[:, 2:8, :]
            best_profile = select_best_profile(xics_top6) # [n_pr, n_cycle]

            # bad_xic by apex
            bad_xic = torch.abs(best_profile.argmax(dim=-1) - expand_dim/2) > 6

            # boundary by best_profile
            box = best_profile > best_profile.max(dim=-1, keepdims=True)[0] * 0.2
            box = box.int()
            box_left = box.argmax(dim=-1)
            box_right = expand_dim - 1 - torch.flip(box, dims=[1]).argmax(dim=-1)

        xics_v.append(xics) # [tol, n_pep, n_ion, n_cycle]
    xics = torch.stack(xics_v, dim=1) # [n_pep, tol, n_ion, n_cycle]

    # find other profile with the help of best_profile
    xics, sas = select_other_profiles(xics, best_profile)

    # interference correction
    xics = interference_correction(xics, best_profile)

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
    xics2 = utils.convert_numba_to_tensor(xics2)  # 14 ions
    xics2 = mask_tensor(xics2, locus_start_v, locus_end_v)
    rts2, xics2 = interp_xics(xics2, rts2, expand_dim)
    xics2 = fxic.gpu_simple_smooth(cuda.to_device(xics2))
    xics2 = utils.convert_numba_to_tensor(xics2)

    xics[bad_xic] = xics2[bad_xic]
    box_left[bad_xic] = 15 # 3-13, 15-64
    box_right[bad_xic] = 50 # 9-13, 50-64

    # boundary
    xics = mask_tensor(xics, box_left, box_right)

    # area not using rts: trapz(xics, rts)
    areas = torch.trapz(xics, dim=-1)

    zeros_idx = (areas == 0) | (sas == 0)
    areas[zeros_idx] = 0.
    sas[zeros_idx] = 0.

    return areas.cpu().numpy(), sas.cpu().numpy()


@profile
def quant_center_ions(df_input, ms):
    df_good = []
    for swath_id in df_input['swath_id'].unique():
        df_swath = df_input[df_input['swath_id'] == swath_id]
        df_swath = df_swath.reset_index(drop=True)

        # ms
        ms1_centroid, ms2_centroid = ms.copy_map_to_gpu(swath_id, centroid=True)

        # in batches
        batch_n = param_g.batch_xic_locus
        for batch_idx, df_batch in df_swath.groupby(df_swath.index // batch_n):
            df_batch = df_batch.reset_index(drop=True)

            # grid search for best profiles
            areas, sas = grid_xic_best(df_batch, ms1_centroid, ms2_centroid)

            # save
            cols = ['score_ion_quant_' + str(i) for i in range(param_g.fg_num + 2)]
            df_batch[cols] = areas
            cols = ['score_ion_sa_' + str(i) for i in range(param_g.fg_num + 2)]
            df_batch[cols] = sas
            df_good.append(df_batch)
        utils.release_gpu_scans(ms1_centroid, ms2_centroid)
    df = pd.concat(df_good, axis=0, ignore_index=True)
    return df
