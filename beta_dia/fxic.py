import math

import numba
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from numba import cuda

from beta_dia import param_g
from beta_dia import utils
from beta_dia.log import Logger

try:
    profile
except NameError:
    profile = lambda x: x

logger = Logger.get_logger()

@cuda.jit(device=True)
def gpu_cal_sa(v):
    '''
    Calculate the sa between V and Gaussian Vector
    '''
    e = 0.000001
    norm_x = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2 + v[3] ** 2 +
                       v[4] ** 2 + v[5] ** 2 + v[6] ** 2) + e

    # y = np.array([0.0044, 0.054, 0.242, 0.399, 0.242, 0.054, 0.0044])
    norm_y = 0.531225
    s = v[0] * 0.0044 + v[1] * 0.054 + v[2] * 0.242 + v[3] * 0.399 + v[
        4] * 0.242 + v[5] * 0.054 + v[6] * 0.0044

    sa = s / (norm_x * norm_y)
    if sa > 1.:
        sa = 1.
    return sa


@cuda.jit
def gpu_sa_gausion_core(block_num, xics, scores, window_points, valids_num):
    '''
    Using share-memory to calculate the sa for each locus
    '''
    # Each block calculates a profile
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    blockdim = cuda.blockDim.x
    if bx >= block_num:
        return

    ions_num = xics.shape[1]
    point_num = xics.shape[2]
    k = bx // ions_num
    xic_idx = bx % ions_num

    # less valid num
    valid_num = valids_num[k]
    if xic_idx > valid_num - 1:
        return

    x = xics[k, xic_idx]
    half = window_points // 2

    # copy to share memory
    # 1h~2000 points，3h~6000 points
    if xics.shape[2] < 500:
        share_xic = cuda.shared.array(500, dtype=numba.float32)
    elif xics.shape[2] < 1500:
        share_xic = cuda.shared.array(1500, dtype=numba.float32)
    else:
        share_xic = cuda.shared.array(5500, dtype=numba.float32)
    # pad for start and end
    if tx == (blockdim - 1):
        share_xic[0] = 0.
    elif tx == (blockdim - 2):
        share_xic[1] = 0.
    elif tx == (blockdim - 3):
        share_xic[2] = 0.
    elif tx == (blockdim - 4):
        share_xic[half + point_num] = 0.
    elif tx == (blockdim - 5):
        share_xic[half + point_num + 1] = 0.
    elif tx == (blockdim - 6):
        share_xic[half + point_num + 2] = 0.

    mean_cols = int(point_num / blockdim)
    if mean_cols < 1:  # less 32
        if tx < point_num:
            share_xic[half + tx] = x[tx]
        cuda.syncthreads()
        # score
        if tx < point_num:
            v = share_xic[tx: (tx + 1 + half + half)]
            score = gpu_cal_sa(v)
            scores[k, xic_idx, tx] = score
    else:
        rest_cols = point_num - mean_cols * blockdim
        for i in range(mean_cols):  # together
            share_xic[half + tx * mean_cols + i] = x[tx * mean_cols + i]
        if tx < rest_cols:  # wind up
            share_xic[half + blockdim * mean_cols + tx] = x[
                blockdim * mean_cols + tx]
        cuda.syncthreads()

        # score
        # together，each thread processes mean_cols
        xx = share_xic[(tx * mean_cols): ((tx + 1) * mean_cols + half + half)]
        for i in range(half, len(xx) - half):
            v = xx[(i - half): (i + half + 1)]
            score = gpu_cal_sa(v)
            scores[k, xic_idx, i - half + tx * mean_cols] = score
        # wind up
        if tx < rest_cols:
            v = share_xic[(blockdim * mean_cols + tx): (
                    blockdim * mean_cols + tx + 1 + half + half)]
            score = gpu_cal_sa(v)
            scores[k, xic_idx, blockdim * mean_cols + tx] = score


@profile
def cal_coelution_by_gaussion(xics, window_points, valids_num):
    '''
    Coelution scores by sliding windows methods
    '''
    valids_num = torch.from_numpy(valids_num).to(param_g.gpu_id)

    # block -- profile
    block_num = xics.shape[0] * xics.shape[1]
    scores = utils.create_cuda_zeros(xics.shape)
    threads_per_block = 32
    gpu_sa_gausion_core[block_num, threads_per_block](block_num,
                                                      xics,
                                                      scores,
                                                      window_points,
                                                      valids_num)
    cuda.synchronize()

    scores = utils.convert_numba_to_tensor(scores)

    scores_raw = 1 - 2 * torch.acos(scores) / np.pi  # [k, f, n]
    scores = torch.sum(scores_raw, dim=1)
    scores = scores / valids_num.view(-1, 1)

    # ends
    scores[:, :3] = 0.
    scores[:, -3:] = 0.
    scores_raw[:, :, :3] = 0.
    scores_raw[:, :, -3:] = 0.

    return scores, scores_raw


# @profile
def gpu_simple_smooth(input_xics):
    '''
    Args:
        input_xics: [n_pep, n_ion, n_cycle]
    '''
    n = input_xics.shape[0] * input_xics.shape[1]
    result_xics = utils.create_cuda_zeros(input_xics.shape)
    threads_per_block = 32  # block -- profile
    blocks_per_grid = n
    gpu_simple_smooth_core[blocks_per_grid, threads_per_block](n,
                                                               input_xics,
                                                               result_xics)
    cuda.synchronize()
    return result_xics


@cuda.jit
def gpu_simple_smooth_core(n, input_xics, output):
    # block -- profile
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x

    # input
    ions_num = input_xics.shape[1]
    k = bx // ions_num
    xic_idx = bx % ions_num
    input_xic = input_xics[k, xic_idx]

    # no share-memory, directly in global memory
    # [0, n-2] --> mean_cols; n-1 --> rest_cols
    blockdim = cuda.blockDim.x
    mean_cols = int(input_xics.shape[2] / blockdim)
    rest_cols = input_xics.shape[2] - mean_cols * (blockdim - 1)

    if tx < blockdim - 1:
        for i in range(mean_cols):
            idx = tx * mean_cols + i
            if idx == 0:
                output[k, xic_idx, idx] = 0.667 * input_xic[idx] + 0.333 * \
                                          input_xic[idx + 1]
            else:
                output[k, xic_idx, idx] = input_xic[idx] * 0.5 + 0.25 * (
                        input_xic[idx + 1] + input_xic[idx - 1])
    else:
        for i in range(rest_cols):
            idx = (blockdim - 1) * mean_cols + i
            if idx == input_xics.shape[2] - 1:
                output[k, xic_idx, idx] = 0.333 * input_xic[idx - 1] + 0.667 * \
                                          input_xic[idx]
            else:
                output[k, xic_idx, idx] = input_xic[idx] * 0.5 + 0.25 * (
                        input_xic[idx + 1] + input_xic[idx - 1])


@cuda.jit(device=True)
def find_maximum(scan_im, scan_mz, scan_height,
                 query_left, query_right,
                 query_im_left, query_im_right):
    '''find the maximum intensity value with tol for query in centroided data'''
    scan_len = len(scan_mz)

    low = 0
    high = scan_len - 1
    best_j = 0
    if scan_mz[low] == query_left:
        best_j = low
    elif scan_mz[high] == query_right:
        best_j = high
    else:
        while high - low > 1:
            mid = (low + high) // 2
            if scan_mz[mid] == query_left:
                best_j = mid
                break
            if scan_mz[mid] < query_left:
                low = mid
            else:
                high = mid
        if best_j == 0:  # no match，high-low=1
            if abs(scan_mz[low] - query_left) < abs(scan_mz[high] - query_left):
                best_j = low
            else:
                best_j = high
    # find first match in list!
    while best_j > 0:
        if scan_mz[best_j - 1] == scan_mz[best_j]:
            best_j = best_j - 1
        else:
            break

    seek_idx = best_j

    best_seek = -1
    y_max = 0
    while seek_idx < scan_len:
        x = scan_mz[seek_idx]
        if x > query_right:
            break
        elif x < query_left:  # exist multiple mz values
            seek_idx += 1
            continue
        else:
            im = scan_im[seek_idx]
            if query_im_left < im < query_im_right:
                y = scan_height[seek_idx]
                if y > y_max:
                    y_max = y
                    best_seek = seek_idx
            seek_idx += 1
    if best_seek > 0:
        im = scan_im[best_seek]
        mz = scan_mz[best_seek]
    else:
        im = -1.
        mz = -1.
    return im, mz, y_max


@cuda.jit
def gpu_extract_xics(
        n,
        cycle_nums,
        idx_start_v,
        ms1_scan_seek_idx,
        ms1_scan_im,
        ms1_scan_mz,
        ms1_scan_height,
        ms2_scan_seek_idx,
        ms2_scan_im,
        ms2_scan_mz,
        ms2_scan_height,
        query_mz_m, ppm_tolerance,
        query_im_v, im_tolerance, ms1_ion_num,
        result_im, result_mz, result_xic, only_xic
):
    # thread -- profile
    thread_idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if thread_idx >= n:
        return

    # pr idx, ion idx
    ions_num = query_mz_m.shape[1]
    k = thread_idx // ions_num
    xic_idx = thread_idx % ions_num

    # params
    query_mz = query_mz_m[k, xic_idx]
    query_mz_left = query_mz * (1. - ppm_tolerance / 1000000.)
    query_mz_right = query_mz * (1. + ppm_tolerance / 1000000.)
    query_im = query_im_v[k]
    query_im_left = query_im - im_tolerance
    query_im_right = query_im + im_tolerance

    ## both for ms1 and ms2
    idx_start = idx_start_v[k]
    idx_end = idx_start + cycle_nums

    if xic_idx < ms1_ion_num:
        scans_seek_idx = ms1_scan_seek_idx
        scans_im = ms1_scan_im
        scans_mz = ms1_scan_mz
        scans_height = ms1_scan_height
    else:
        scans_seek_idx = ms2_scan_seek_idx
        scans_im = ms2_scan_im
        scans_mz = ms2_scan_mz
        scans_height = ms2_scan_height

    for cycle_idx, scan_idx in enumerate(range(idx_start, idx_end)):
        start = scans_seek_idx[scan_idx]
        end = scans_seek_idx[scan_idx + 1]
        scan_im = scans_im[start: end]
        scan_mz = scans_mz[start: end]
        scan_height = scans_height[start: end]

        im, mz, y_max = find_maximum(
            scan_im, scan_mz, scan_height,
            query_mz_left, query_mz_right,
            query_im_left, query_im_right
        )

        if not only_xic:
            result_im[k, xic_idx, cycle_idx] = im
            result_mz[k, xic_idx, cycle_idx] = mz
        result_xic[k, xic_idx, cycle_idx] = y_max


@profile
def extract_xics(df,
                 map_gpu_ms1,
                 map_gpu_ms2,
                 ppm_tolerance,
                 im_tolerance,
                 rt_tolerance=None,
                 cycle_num=None,
                 scope='center',
                 only_xic=False,
                 by_pred=True):
    '''
    Extrac XICs from centroid ms data.
    Args:
        df:
        map_gpu_ms1:
        map_gpu_ms2:
        ppm_tolerance:
        im_tolerance:
        rt_tolerance:
        cycle_num: either rt_tolerance or cycle_num
        scope: which ions to consider
        only_xic:
        by_pred: use measure_im or pred_im
    Returns:
        cycles_idx, rts, ims, mzs, xics
    '''
    scan_rts = map_gpu_ms1['scan_rts']
    cycle_total = len(scan_rts)
    biggest_rt = scan_rts[-1]

    # rt_range -- cycle start
    cycle_time = np.mean(np.diff(scan_rts))
    if (rt_tolerance is None) and (cycle_num is None):  # all rt_range
        idx_start_v = np.zeros(len(df), dtype=np.int32)
        cycle_num = cycle_total
        # cycle -- rt
        result_cycle_idx = np.broadcast_to(np.arange(cycle_total), (len(df), cycle_total))
        result_rts = np.broadcast_to(scan_rts, (len(df), cycle_total))
    elif rt_tolerance is not None:
        cycle_num = int(rt_tolerance * 2 / cycle_time)
        if cycle_num > cycle_total:
            cycle_num = cycle_total
        rt_range_low = df['pred_rt'].values - rt_tolerance
        idx_start_v = rt_range_low / biggest_rt
        idx_start_v = (idx_start_v * (len(scan_rts) - 1)).astype(np.int32)
        idx_start_v[idx_start_v < 0] = 0
        idx_start_max = cycle_total - cycle_num
        idx_start_v[idx_start_v > idx_start_max] = idx_start_max
        # cycle -- rt
        cycle_idx = np.arange(cycle_num) + idx_start_v[:, None]
        result_cycle_idx = np.arange(cycle_total)[cycle_idx]
        result_rts = scan_rts[cycle_idx]
    elif cycle_num is not None:
        cycle_total = len(map_gpu_ms1['scan_rts'])
        idx_start_v = df['locus'].values - int(cycle_num / 2)
        idx_start_v[idx_start_v < 0] = 0
        idx_start_max = cycle_total - cycle_num
        idx_start_v[idx_start_v > idx_start_max] = idx_start_max
        # cycle -- rt
        cycle_idx = np.arange(cycle_num) + idx_start_v[:, None]
        result_cycle_idx = np.arange(cycle_total)[cycle_idx]
        result_rts = scan_rts[cycle_idx]
    else:
        assert 1 > 2, 'Set either rt_tolerance or cycle_tolerance!'

    # params
    if scope == 'center':
        query_mz_ms1 = df[['pr_mz', 'pr_mz']].values
        fg_mz_cols = ['fg_mz_' + str(i) for i in range(param_g.fg_num)]
        query_mz_ms2 = df[fg_mz_cols].values
        query_mz_m = np.concatenate([query_mz_ms1, query_mz_ms2], axis=1)
        ms1_ion_num = 1
    elif scope == 'big':
        ms1_cols = ['pr_mz_left', 'pr_mz', 'pr_mz_1H', 'pr_mz_2H',
                    'pr_mz_left', 'pr_mz', 'pr_mz_1H', 'pr_mz_2H']  # unfrag
        ms1 = df[ms1_cols].values
        cols_left = ['fg_mz_left_' + str(i) for i in range(param_g.fg_num)]
        left = df[cols_left].values
        cols_center = ['fg_mz_' + str(i) for i in range(param_g.fg_num)]
        center = df[cols_center].values
        cols_1H = ['fg_mz_1H_' + str(i) for i in range(param_g.fg_num)]
        fg_1H = df[cols_1H].values
        cols_2H = ['fg_mz_2H_' + str(i) for i in range(param_g.fg_num)]
        fg_2H = df[cols_2H].values
        query_mz_m = np.concatenate([ms1, left, center, fg_1H, fg_2H], axis=1)
        ms1_ion_num = 4
    elif scope == 'top6':
        cols_center = ['fg_mz_' + str(i) for i in range(param_g.fg_num)]
        query_mz_m = np.ascontiguousarray(df[cols_center].values[:, :6])
        ms1_ion_num = 0

    if by_pred:
        query_im_v = df['pred_im'].values
    else:
        query_im_v = df['measure_im'].values

    # GPU
    ions_num = query_mz_m.shape[1]
    if only_xic:
        result_im = cuda.device_array((1, 1, 1), dtype=np.float32)
        result_mz = cuda.device_array((1, 1, 1), dtype=np.float32)
    else:
        result_im = cuda.device_array(
            (len(df), ions_num, cycle_num), dtype=np.float32
        )
        result_mz = cuda.device_array(
            (len(df), ions_num, cycle_num), dtype=np.float32
        )
    result_xic = cuda.device_array(
        (len(df), ions_num, cycle_num), dtype=np.float32
    )
    idx_start_v = cuda.to_device(idx_start_v)
    query_mz_m = cuda.to_device(query_mz_m)
    query_im_v = cuda.to_device(query_im_v)

    # kernel func, each thread is for a profile of an ion
    k = df.shape[0]
    n = k * ions_num
    threads_per_block = 512
    blocks_per_grid = math.ceil(n / threads_per_block)
    gpu_extract_xics[blocks_per_grid, threads_per_block](
        n,
        cycle_num,
        idx_start_v,
        map_gpu_ms1['scan_seek_idx'],
        map_gpu_ms1['scan_im'],
        map_gpu_ms1['scan_mz'],
        map_gpu_ms1['scan_height'],
        map_gpu_ms2['scan_seek_idx'],
        map_gpu_ms2['scan_im'],
        map_gpu_ms2['scan_mz'],
        map_gpu_ms2['scan_height'],
        query_mz_m, ppm_tolerance,
        query_im_v, im_tolerance, ms1_ion_num,
        result_im, result_mz, result_xic, only_xic
    )
    cuda.synchronize()

    if only_xic:
        return (result_cycle_idx, result_rts, result_xic)
    else:
        if scope != 'big':
            return (
                result_cycle_idx,
                result_rts,
                result_im.copy_to_host(),
                result_mz.copy_to_host(),
                result_xic
            )
        else: # order on [left, center, 1H, 2H]
            result_im = result_im.copy_to_host()
            result_mz = result_mz.copy_to_host()
            result_xic = utils.convert_numba_to_tensor(result_xic)
            ims_v, mzs_v, xics_v = [], [], []
            for i in range(4):
                idx = [i, i + 4] + list(range(8 + i * 12, 20 + i * 12))
                ims_v.append(result_im[:, idx])
                mzs_v.append(result_mz[:, idx])
                xics_v.append(result_xic[:, idx])
            return (result_cycle_idx, result_rts, ims_v, mzs_v, xics_v)


@profile
def cal_measure_im(locus_ims, locus_sas, good_cut=0.5):
    '''
    Calculate the measure_im for each locus, weighting with the sa values.
    Args:
        locus_ims: [n_locus, n_ion]
        locus_sas: [n_locus, n_ion]
        good_cut: only considering the ion with good_cut threshold

    Returns:
        [n_locus]
    '''
    condition1 = (locus_ims <= 0.)
    condition2 = (locus_sas < locus_sas.max(axis=-1, keepdims=True) * good_cut)
    bad_idx = condition1 | condition2

    locus_ims[bad_idx] = 0.
    locus_sas[bad_idx] = 0.

    locus_sas += 1e-7
    locus_im = np.average(locus_ims, weights=locus_sas, axis=-1)

    # assert locus_im.min() > 0.2
    # assert locus_im.max() < 2.

    return locus_im


def reserve_sa_maximum(x):
    '''
    If x > x-1 and x > x+1, x is local maximum will be saved. If not, assign 0
    Args:
        x (2D Tensor): [n_pep, n_cycle]

    Returns:
        [n_pep, n_cycle]
    '''
    x_pad = F.pad(x, (1, 1))
    idx = (x_pad[:, 1:-1] > x_pad[:, 2:]) & (x_pad[:, 1:-1] > x_pad[:, 0:-2])
    x[~idx] = 0
    return x


def screen_locus_by_sa(scores_sa, top_sa_cut):
    '''
    Screen multi locus of a pr that satify: local maximum, quantile1, quantile2
    Args:
        scores_sa: [n_pep, n_cycle]
        top_sa_cut: quantile threshold on sa level

    Returns:
        scores_sa: bad points is assigned zero
    '''
    median_values = scores_sa.quantile(top_sa_cut, dim=1, keepdim=True)
    rowmax_values = scores_sa.amax(dim=1, keepdim=True)

    # local maximum
    scores_sa = reserve_sa_maximum(scores_sa)

    # screen
    condition1 = (scores_sa / rowmax_values) < top_sa_cut
    condition2 = scores_sa < median_values
    bad_idx = condition1 | condition2
    scores_sa[bad_idx] = 0.

    return scores_sa


@profile
def screen_locus_by_deep(df_batch, locus_num, top_deep_q):
    '''
    Screen locuses of a pr by deep scores.
    Args:
        df_batch: n_pep*n_locus rows
        top_deep_q: threshold for deep_x / deep_max

    Returns:
        df_batch: less rows
    '''
    group_size_cumsum = np.concatenate([[0], np.cumsum(locus_num)])
    group_rank_deep = utils.cal_group_rank(
        df_batch['seek_score_deep'].values, group_size_cumsum
    )
    group_rank_x = utils.cal_group_rank(
        df_batch['seek_score_sa_x_deep'].values, group_size_cumsum
    )

    # screen by top-n
    idx = (group_rank_deep <= 1) | (group_rank_x <= 1)

    # screen by ratio
    # group_max = df_batch.groupby('pr_id')['seek_score_sa_x_deep'].transform(
    #     'max')
    # ratios = df_batch['seek_score_sa_x_deep'] / group_max
    # idx = ratios > top_deep_q

    df_batch = df_batch[idx].reset_index(drop=True)
    return df_batch


def concat_nonzero_locus(locus, scores_sa, scores_sa_m):
    '''
    After screening locus by sa, sa_input has much zero values. Selecting and
    concat the nonzero values to vectors.
    Args:
        locus: [n_pep, n_locus]
        scores_sa: [n_pep, n_locus]
        scores_sa_m: [n_pep, n_ion, n_locus]

    Returns:
        locus_v, locus_num/pr, sa_1d
    '''
    good_idx = scores_sa > 0
    locus_sa_v = scores_sa[good_idx].cpu().numpy()
    locus_sas = scores_sa_m.transpose(1, 2)[good_idx].cpu().numpy()

    good_idx = good_idx.cpu().numpy()
    locus_v = locus[good_idx]
    locus_num = good_idx.sum(axis=1)

    assert len(locus_v) == len(locus_sas) == locus_num.sum()

    return locus_v, locus_num, locus_sa_v, locus_sas


def estimate_xic_boundary(xics, sa_gausion_m):
    '''
    Exstimate the boundary of an elution group in cycles.
    Args:
        xics: [n_pep, n_ion, 13]
        sa_gausion_m: [n_pep, n_ion]

    Returns:
        left_idx_1d, right_idx_1d
    '''
    center_idx = int(xics.shape[-1] / 2)
    sa_sum = sa_gausion_m.sum(dim=1)

    # find valley
    x_pad = F.pad(xics, (1, 1))
    left_condition1 = x_pad[:, :, 1:-1] < x_pad[:, :, 2:]
    left_condition2 = x_pad[:, :, 1:-1] <= x_pad[:, :, 0:-2]

    right_condition1 = x_pad[:, :, 1:-1] <= x_pad[:, :, 2:]
    right_condition2 = x_pad[:, :, 1:-1] < x_pad[:, :, 0:-2]

    # left
    condition = (left_condition1 & left_condition2)
    condition = condition[:, :, :center_idx].int()
    condition = condition.flip(2)
    left_idx = torch.argmax(condition, dim=2)  # first left valley
    left_idx = center_idx - 1 - left_idx

    no_valley_idx = condition.sum(dim=2) == 0
    left_idx[no_valley_idx] = center_idx - 3  # no valley, set to half of 7

    left_idx = (left_idx * sa_gausion_m).sum(dim=1) / (sa_sum + 1e-7)
    left_idx = torch.round(left_idx)  # or ceil

    # right
    condition = (right_condition1 & right_condition2)
    condition = condition[:, :, (center_idx + 1):].int()
    right_idx = torch.argmax(condition, dim=2)  # first right valley
    right_idx = center_idx + 1 + right_idx

    no_valley_idx = condition.sum(dim=2) == 0
    right_idx[no_valley_idx] = center_idx + 3  # no valley, set to half of 7

    right_idx = (right_idx * sa_gausion_m).sum(dim=1) / (sa_sum + 1e-7)
    right_idx = torch.round(right_idx)  # or floor

    return left_idx.int().cpu().numpy(), right_idx.int().cpu().numpy()


from itertools import product

def grid_xic_best(df_batch, ms1_centroid, ms2_centroid):
    locus_start_v = df_batch['score_elute_span_left'].values
    locus_end_v = df_batch['score_elute_span_right'].values

    tol_ppm_v = [20., 16., 12., 8., 4.]
    tol_im_v = [0.03, 0.02, 0.01]
    grid_params = list(product(tol_ppm_v, tol_im_v))

    xics_v = []
    expand_dim = 64
    for search_i, (tol_ppm, tol_im) in enumerate(grid_params):
        _, rts, _, _, xics = extract_xics(
            df_batch,
            ms1_centroid,
            ms2_centroid,
            im_tolerance=tol_im,
            ppm_tolerance=tol_ppm,
            cycle_num=13,
            by_pred=False
        )
        xics = xics.copy_to_host()
        mask1 = np.arange(xics.shape[2]) >= locus_start_v[:, None, None]
        mask2 = np.arange(xics.shape[2]) <= locus_end_v[:, None, None]
        xics = xics * mask1 * mask2
        rts, xics = utils.interp_xics(xics, rts, expand_dim)
        xics = gpu_simple_smooth(cuda.to_device(xics))
        xics = xics.copy_to_host()

        # find best profile
        if search_i == 0:
            sas = np.array(list(map(utils.cross_cos, xics)))
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
    sas = sas.max(axis=1)

    return sas


def update_sa_by_grid(df, ms):
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
            sas = grid_xic_best(df_batch, ms1_centroid, ms2_centroid)

            cols_center = ['score_center_elution_' + str(i) for i in range(14)]
            df_batch[cols_center] = sas
            df_good.append(df_batch)

        utils.release_gpu_scans(ms1_centroid, ms2_centroid)

    df = pd.concat(df_good, axis=0, ignore_index=True)
    return df
