import math
from pathlib import Path

import numpy as np
import torch
from numba import cuda

from beta_dia import models
from beta_dia import param_g
from beta_dia import utils
from beta_dia.utils import create_cuda_zeros

try:
    profile
except:
    profile = lambda x: x

def load_models(dir_center=None, dir_big=None):
    channels = 2 + param_g.fg_num
    model_center = load_model_center(dir_center, channels)

    channels = 4*(2 + param_g.fg_num)
    model_big = load_model_big(dir_big, channels)

    # import torch._dynamo
    # torch._dynamo.config.suppress_errors = True
    # model_center = torch.compile(model_center, mode='reduce-overhead')
    # model_big = torch.compile(model_big, mode='reduce-overhead ')

    return model_center, model_big


def load_model_big(dir_model, channels):
    model = models.DeepMap(channels)
    device = param_g.gpu_id
    if dir_model is None:
        pt_path = Path(__file__).resolve().parent/'pretrained'/'deepbig_ys_fast.pt'
        model.load_state_dict(torch.load(pt_path, map_location=device))
    else:
        model.load_state_dict(torch.load(dir_model, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def load_model_center(dir_model, channels):
    model = models.DeepMap(channels)
    device = param_g.gpu_id
    if dir_model is None:
        pt_path = Path(__file__).resolve().parent/'pretrained'/'deepcenter_ys_fast.pt'
        model.load_state_dict(torch.load(pt_path, map_location=device))
    else:
        model.load_state_dict(torch.load(dir_model, map_location=device))
    model = model.to(device)
    model.eval()
    return model


@cuda.jit(device=True)
def find_first_index(scan_mz, query_left, query_right):
    '''
    Find first index that match the query value
    Args:
        scan_mz: ms data of a cycle with m/z ascending order
        query_left:
        query_right:

    Returns:
        index
    '''
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
        if best_j == 0:  # on match，high-low=1
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

    return best_j


@cuda.jit
def gpu_bin_map(
        n,
        cycle_num,
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
        query_im_v, im_tolerance, im_gap,
        result_maps,
        ms1_ion_num
):
    '''
    Each thread generates maps of an elution group
    '''
    thread_idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if thread_idx >= n:
        return

    # pr idx, ion idx
    ions_num = query_mz_m.shape[1]
    k = thread_idx // ions_num
    ion_idx = thread_idx % ions_num

    # params
    query_mz = query_mz_m[k, ion_idx]
    query_mz_left = query_mz * (1. - ppm_tolerance / 1000000.)
    query_mz_right = query_mz * (1. + ppm_tolerance / 1000000.)
    query_im = query_im_v[k]
    query_im_left = query_im - im_tolerance
    query_im_right = query_im + im_tolerance
    im_base = query_im - im_tolerance

    # both for ms1 and ms2
    idx_start = idx_start_v[k]
    idx_end = idx_start + cycle_num

    if ion_idx < ms1_ion_num:
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
        scan_len = end - start
        scan_im = scans_im[start: end]
        scan_mz = scans_mz[start: end]
        scan_height = scans_height[start: end]

        seek = find_first_index(scan_mz, query_mz_left, query_mz_right)

        while seek < scan_len:
            mz = scan_mz[seek]
            if mz > query_mz_right:
                break
            elif mz < query_mz_left:  # exist multiple mz values
                seek += 1
                continue
            else:
                im = scan_im[seek]
                if query_im_left < im < query_im_right:
                    y = scan_height[seek]
                    im_idx = int((im - im_base) / im_gap)
                    y_map_curr = result_maps[k, ion_idx, cycle_idx, im_idx]
                    if y > y_map_curr:
                        result_maps[k, ion_idx, cycle_idx, im_idx] = y
                seek += 1


@cuda.jit
def gpu_bin_maps(
        n, locus_num,
        cycle_num,
        idx_start_m,
        ms1_scan_seek_idx,
        ms1_scan_im,
        ms1_scan_mz,
        ms1_scan_height,
        ms2_scan_seek_idx,
        ms2_scan_im,
        ms2_scan_mz,
        ms2_scan_height,
        query_mz_m, tol_ppm,
        query_im_v, tol_im_map, im_gap,
        result_maps,
        ms1_ion_num,
):
    '''
    maps: [n_pr, n_locus, n_ion, n_cycle, n_im_bin]
    Each thread generates maps for multi elution groups of a pr
    '''
    thread_idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if thread_idx >= n:
        return

    # pr idx, ion idx
    ions_num = query_mz_m.shape[1]
    locus_per_peptide = locus_num * ions_num
    k = thread_idx // locus_per_peptide
    locus = thread_idx % locus_per_peptide // ions_num
    ion_idx = thread_idx % locus_per_peptide % ions_num

    # params
    query_mz = query_mz_m[k, ion_idx]
    query_mz_left = query_mz * (1. - tol_ppm / 1000000.)
    query_mz_right = query_mz * (1. + tol_ppm / 1000000.)
    query_im = query_im_v[k]
    query_im_left = query_im - tol_im_map
    query_im_right = query_im + tol_im_map
    im_base = query_im - tol_im_map

    ## both for ms1 and ms2
    idx_start = idx_start_m[k, locus]
    idx_end = idx_start + cycle_num

    if ion_idx < ms1_ion_num:
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
        scan_len = end - start
        scan_im = scans_im[start: end]
        scan_mz = scans_mz[start: end]
        scan_height = scans_height[start: end]

        seek = find_first_index(scan_mz, query_mz_left, query_mz_right)
        while seek < scan_len:
            x = scan_mz[seek]
            if x > query_mz_right:
                break
            elif x < query_mz_left:  # exist multi mz values
                seek += 1
                continue
            else:
                im = scan_im[seek]
                if query_im_left < im < query_im_right:
                    y = scan_height[seek]
                    im_idx = int((im - im_base) / im_gap)
                    y_curr = result_maps[k, locus, ion_idx, cycle_idx, im_idx]
                    if y > y_curr:
                        result_maps[k, locus, ion_idx, cycle_idx, im_idx] = y
                seek += 1


@profile
def extract_maps(df_batch,
                 idx_start_m,
                 locus_num,
                 cycle_num,
                 map_im_size,
                 map_gpu_ms1,
                 map_gpu_ms2,
                 tol_ppm,
                 tol_im_map,
                 im_gap,
                 neutron_num
                 ):
    '''
    Extrac maps for multi elution groups of a pr
    Args:
        df_batch: provide pr info
        idx_start_m: cycle start index
        locus_num:
        cycle_num:
        map_im_size:
        map_gpu_ms1:
        map_gpu_ms2:
        tol_ppm:
        tol_im_map:
        im_gap:
        neutron_num:

    Returns:
        Maps (Tensor)
    '''
    batch_size = len(df_batch)

    idx_start = idx_start_m[df_batch.index]

    # params
    if neutron_num == -1:
        query_mz_ms1 = df_batch['pr_mz_left'].values
        query_mz_ms1 = np.tile(query_mz_ms1, (2, 1)).T
        query_mz_ms2 = np.array(df_batch['fg_mz_left'].values.tolist())
        query_mz_m = np.concatenate([query_mz_ms1, query_mz_ms2], axis=1)
        ms1_ion_num = 1
    elif neutron_num == 0:
        query_mz_ms1 = df_batch['pr_mz'].values
        query_mz_ms1 = np.tile(query_mz_ms1, (2, 1)).T
        cols_center = ['fg_mz_' + str(i) for i in range(param_g.fg_num)]
        query_mz_ms2 = df_batch[cols_center].values
        query_mz_m = np.concatenate([query_mz_ms1, query_mz_ms2], axis=1)
        ms1_ion_num = 1
    elif neutron_num == 1:
        query_mz_ms1 = df_batch['pr_mz_1H'].values
        query_mz_ms1 = np.tile(query_mz_ms1, (2, 1)).T
        cols_1H = ['fg_mz_1H_' + str(i) for i in range(param_g.fg_num)]
        query_mz_ms2 = df_batch[cols_1H].values
        query_mz_m = np.concatenate([query_mz_ms1, query_mz_ms2], axis=1)
        ms1_ion_num = 1
    elif neutron_num == 2:
        query_mz_ms1 = df_batch['pr_mz_2H'].values
        query_mz_ms1 = np.tile(query_mz_ms1, (2, 1)).T
        cols_2H = ['fg_mz_2H_' + str(i) for i in range(param_g.fg_num)]
        query_mz_ms2 = df_batch[cols_2H].values
        query_mz_m = np.concatenate([query_mz_ms1, query_mz_ms2], axis=1)
        ms1_ion_num = 1
    elif neutron_num > 2:  # total
        ms1_cols = ['pr_mz_left', 'pr_mz', 'pr_mz_1H', 'pr_mz_2H',
                    'pr_mz_left', 'pr_mz', 'pr_mz_1H', 'pr_mz_2H'] # unfrag
        ms1 = df_batch[ms1_cols].values
        cols_left = ['fg_mz_left_' + str(i) for i in range(param_g.fg_num)]
        left = df_batch[cols_left].values
        cols_center = ['fg_mz_' + str(i) for i in range(param_g.fg_num)]
        center = df_batch[cols_center].values
        cols_1H = ['fg_mz_1H_' + str(i) for i in range(param_g.fg_num)]
        fg_1H = df_batch[cols_1H].values
        cols_2H = ['fg_mz_2H_' + str(i) for i in range(param_g.fg_num)]
        fg_2H = df_batch[cols_2H].values
        query_mz_m = np.concatenate([ms1, left, center, fg_1H, fg_2H], axis=1)
        ms1_ion_num = 4
    else:
        assert 0 > 1, 'neutron_num has to be [-1, 1, 2, >2]'

    query_im_v = df_batch['measure_im'].values

    # GPU
    idx_start = cuda.to_device(idx_start)
    query_mz_m = cuda.to_device(query_mz_m)
    query_im_v = cuda.to_device(query_im_v)
    result_maps = create_cuda_zeros((batch_size,
                                     locus_num,
                                     query_mz_m.shape[1],
                                     cycle_num,
                                     map_im_size))
    # kernel func, each thread generates maps for a pr
    k = batch_size
    n = k * locus_num * query_mz_m.shape[1]
    threads_per_block = 512
    blocks_per_grid = math.ceil(n / threads_per_block)
    gpu_bin_maps[blocks_per_grid, threads_per_block](
        n, locus_num,
        cycle_num,
        idx_start,
        map_gpu_ms1['scan_seek_idx'],
        map_gpu_ms1['scan_im'],
        map_gpu_ms1['scan_mz'],
        map_gpu_ms1['scan_height'],
        map_gpu_ms2['scan_seek_idx'],
        map_gpu_ms2['scan_im'],
        map_gpu_ms2['scan_mz'],
        map_gpu_ms2['scan_height'],
        query_mz_m, tol_ppm,
        query_im_v, tol_im_map, im_gap,
        result_maps,
        ms1_ion_num,
    )
    cuda.synchronize()

    result_maps = utils.convert_numba_to_tensor(result_maps)
    return result_maps


@profile
def scoring_maps(
        model,
        df_input,
        map_gpu_ms1,
        map_gpu_ms2,
        cycle_num,
        map_im_gap, map_im_dim,
        ppm_tolerance,
        im_tolerance,
        neutron_num,
        return_feature=True
):
    '''
    Extract and scoring the Maps-14 for multi elution groups of a pr
    Args:
        model: DeepProfile-14/DeepMap-14
        df_input:
        map_gpu_ms1:
        map_gpu_ms2:
        cycle_num:
        map_im_gap:
        map_im_dim:
        ppm_tolerance:
        im_tolerance:
        neutron_num:
        return_feature: bool

    Returns:
        pred and features
    '''
    # locus
    locus_m = df_input['locus'].values.reshape(-1, 1)
    locus_num = locus_m.shape[1]

    # cycle start and end
    cycle_total = len(map_gpu_ms1['scan_rts'])
    idx_start_m = locus_m - int((cycle_num - 1) / 2)
    idx_start_m[idx_start_m < 0] = 0
    idx_start_max = cycle_total - cycle_num
    idx_start_m[idx_start_m > idx_start_max] = idx_start_max

    # in batches
    feature_v, pred_v = [], []
    batch_num = param_g.batch_deep_center
    for batch_idx, df_batch in df_input.groupby(df_input.index // batch_num):
        maps = extract_maps(df_batch,
                            idx_start_m,
                            locus_num,
                            cycle_num,
                            map_im_dim,
                            map_gpu_ms1,
                            map_gpu_ms2,
                            ppm_tolerance,
                            im_tolerance, map_im_gap,
                            neutron_num=neutron_num)
        # maps: [k, locus, 2+fg_num, map_cycle_dim, map_im_dim]
        maps = maps.view(maps.shape[0] * maps.shape[1],
                         maps.shape[2],
                         maps.shape[3],
                         maps.shape[4])
        # valid ion nums
        non_fg_num = maps.shape[1] - param_g.fg_num
        valid_ion_nums = non_fg_num + df_batch['fg_num'].values
        valid_ion_nums = torch.from_numpy(
            np.repeat(valid_ion_nums, locus_num)).long().to(param_g.gpu_id)
        with torch.no_grad():
            # with torch.cuda.amp.autocast():
            feature, pred = model(maps, valid_ion_nums)
        torch.cuda.synchronize()  # for profile

        pred = torch.softmax(pred, 1)
        pred = pred[:, 1].view(len(df_batch), locus_num)
        pred_v.append(pred)

        if return_feature:
            feature = feature.view(len(df_batch), locus_num, -1)
            feature = feature.cpu()
            feature = feature.numpy()
            feature_v.append(feature)

    pred = torch.cat(pred_v).to(dtype=torch.float32) # torch autocast to 16
    if return_feature:
        feature = np.vstack(feature_v)
    else:
        feature = None

    return pred, feature


@profile
def extract_scoring_big(
        model_center,
        model_big,
        df_input,
        map_gpu_ms1,
        map_gpu_ms2,
        cycle_num,
        map_im_gap, map_im_dim,
        ppm_tolerance,
        im_tolerance,
):
    '''
    Extrac and scoring Maps for elution groups-56
    Args:
        model_center: Scoring elution groups-14
        model_big: Scoring elution groups-56
        df_input:
        map_gpu_ms1:
        map_gpu_ms2:
        cycle_num:
        map_im_gap:
        map_im_dim:
        ppm_tolerance:
        im_tolerance:

    Returns:
        pred_v, feature_v: [14-left, 14-center, 14-1H, 14-2H, 56-total]
    '''
    # locus
    locus_v = df_input['locus'].values

    # cycle start and end
    cycle_total = len(map_gpu_ms1['scan_rts'])
    idx_start_v = locus_v - int((cycle_num - 1) / 2)
    idx_start_v[idx_start_v < 0] = 0
    idx_start_max = cycle_total - cycle_num
    idx_start_v[idx_start_v > idx_start_max] = idx_start_max

    # params
    ms1_cols = ['pr_mz_left', 'pr_mz', 'pr_mz_1H', 'pr_mz_2H',
                'pr_mz_left', 'pr_mz', 'pr_mz_1H', 'pr_mz_2H']  # unfrag
    ms1 = df_input[ms1_cols].values
    cols_left = ['fg_mz_left_' + str(i) for i in range(param_g.fg_num)]
    left = df_input[cols_left].values
    cols_center = ['fg_mz_' + str(i) for i in range(param_g.fg_num)]
    center = df_input[cols_center].values
    cols_1H = ['fg_mz_1H_' + str(i) for i in range(param_g.fg_num)]
    fg_1H = df_input[cols_1H].values
    cols_2H = ['fg_mz_2H_' + str(i) for i in range(param_g.fg_num)]
    fg_2H = df_input[cols_2H].values
    query_mz_m = np.concatenate([ms1, left, center, fg_1H, fg_2H], axis=1)
    ms1_ion_num = 4

    query_im_v = df_input['measure_im'].values

    # cuda input
    idx_start_v = cuda.to_device(idx_start_v)
    query_mz_m = cuda.to_device(query_mz_m)
    query_im_v = cuda.to_device(query_im_v)

    # cuda output
    n = len(df_input)
    ions_num = query_mz_m.shape[1]
    maps = create_cuda_zeros((n, ions_num, cycle_num, map_im_dim))

    # kernel func, each thread for a elution groups of a pr
    thread_num = n * ions_num
    threads_per_block = 256
    blocks_per_grid = math.ceil(thread_num / threads_per_block)
    gpu_bin_map[blocks_per_grid, threads_per_block](
        thread_num,
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
        query_im_v, im_tolerance, map_im_gap,
        maps,
        ms1_ion_num
    )
    cuda.synchronize()

    # -1H, center, +H, +2H, total
    maps = utils.convert_numba_to_tensor(maps)

    pred_v, feature_v = [], []
    for i in range(5):
        if i != 4:
            idx = [i, i+4] + list(range(8+i*12, 20+i*12))
            valid_ion_nums = 2 + df_input['fg_num'].values
            model = model_center
        else:
            idx = list(range(56))
            valid_ion_nums = 4 * (2 + df_input['fg_num'].values)
            model = model_big
        maps_sub = maps[:, idx]
        valid_ion_nums = torch.from_numpy(valid_ion_nums).long().to(param_g.gpu_id)
        with torch.no_grad():
            # with torch.cuda.amp.autocast():
            feature, pred = model(maps_sub, valid_ion_nums)
        torch.cuda.synchronize()
        pred = torch.softmax(pred, 1)
        pred = pred[:, 1].cpu().numpy().astype(np.float32)
        feature = feature.cpu().numpy()
        pred_v.append(pred)
        feature_v.append(feature)

    return pred_v, feature_v
