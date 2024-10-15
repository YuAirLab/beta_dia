import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from numba import cuda
from numba import jit, prange

from beta_dia import alphatims
from beta_dia.alphatims import bruker
from beta_dia.log import Logger

try:
    profile
except:
    profile = lambda x: x

logger = Logger.get_logger()

@jit(nopython=True, nogil=True)
def numba_index_by_bool(idx, ims, mzs, heights):
    result_len = np.sum(idx)
    result_ims = np.empty(result_len, dtype=ims.dtype)
    result_mzs = np.empty(result_len, dtype=mzs.dtype)
    result_heights = np.empty(result_len, dtype=heights.dtype)
    # idx_cumsum = np.cumsum(idx)
    write_idx = 0
    for i in range(len(idx)):
        if idx[i]:
            # write_idx = idx_cumsum[i] - 1
            result_ims[write_idx] = ims[i]
            result_mzs[write_idx] = mzs[i]
            result_heights[write_idx] = heights[i]
            write_idx += 1
    return result_ims, result_mzs, result_heights


@jit(nopython=True, nogil=True, parallel=True)
def numba_paral_repeat(x, y):
    result = np.empty(y[-1], dtype=x.dtype)
    for i in prange(len(x)):
        value = x[i]
        start = y[i]
        end = y[i + 1]
        for ii in range(start, end):
            result[ii] = value
    return result


@jit(nopython=True, nogil=True, parallel=True)
def numba_paral_sum(select_id, cumlen):
    result = np.empty(len(cumlen), dtype=np.int64)

    for i in prange(len(cumlen)):
        if i == 0:
            start = 0
            end = cumlen[i]
        else:
            start = cumlen[i - 1]
            end = cumlen[i]

        value = np.sum(select_id[start: end])
        result[i] = value

    return result


@jit(nopython=True, nogil=True, parallel=True)
def numba_paral_sort(all_tof, all_push, all_height, cumlen):
    result_tof = np.empty(len(all_tof), dtype=all_tof.dtype)
    result_push = np.empty(len(all_push), dtype=all_push.dtype)
    result_height = np.empty(len(all_height), dtype=all_height.dtype)

    for i in prange(len(cumlen)):
        if i == 0:
            start = 0
            end = cumlen[i]
        else:
            start = cumlen[i - 1]
            end = cumlen[i]

        tof = all_tof[start: end]
        push = all_push[start: end]
        height = all_height[start: end]
        idx = np.argsort(tof)

        result_tof[start: end] = tof[idx]
        result_push[start: end] = push[idx]
        result_height[start: end] = height[idx]

    return result_tof, result_push, result_height


@jit(nopython=True, nogil=True, parallel=True)
def numba_paral_centroid(
        all_tof, all_push, all_height,
        tol_tof_sum, tol_tof_suppression, tol_push, cumlen
):
    # assert all_tof.dtype == np.uint32 # progressive increase
    # assert all_push.dtype == np.int16 # exist subtraction
    # assert all_height.dtype == np.uint16 # maybe overflow

    all_height_summed = np.zeros_like(all_height, dtype=np.uint32)
    all_height_suppressed = np.ones_like(all_height)

    # first summed surrounding
    for ms_i in prange(len(cumlen)):
        if ms_i == 0:
            start = 0
            end = cumlen[ms_i]
        else:
            start = cumlen[ms_i - 1]
            end = cumlen[ms_i]

        tof = all_tof[start : end]
        push = all_push[start : end]
        height = all_height[start : end]

        if np.sum(height) == 0:
            continue

        # sum
        for i in range(len(tof)):
            tof_i = tof[i]
            push_i = push[i]
            height_i = height[i]
            all_height_summed[start + i] += height_i
            for ii in range(i+1, len(tof)):
                tof_ii = tof[ii]
                push_ii = push[ii]
                height_ii = height[ii]
                delta_tof = tof_ii - tof_i
                delta_push = abs(push_ii - push_i)
                if delta_tof > tol_tof_sum:
                    break
                if delta_push > tol_push:
                    continue
                all_height_summed[start + ii] += height_i
                all_height_summed[start + i] += height_ii

        # suppression
        for i in range(len(tof)):
            tof_i = tof[i]
            push_i = push[i]
            height_i = all_height_summed[start + i]
            for ii in range(i + 1, len(tof)):
                tof_ii = tof[ii]
                push_ii = push[ii]
                height_ii = all_height_summed[start + ii]
                delta_tof = tof_ii - tof_i
                delta_push = abs(push_ii - push_i)
                if delta_tof > tol_tof_suppression:
                    break
                if delta_push > tol_push:
                    continue
                if height_ii > height_i:
                    all_height_suppressed[start + i] = 0.
                elif height_ii < height_i:
                    all_height_suppressed[start + ii] = 0.
                else:
                    height_raw_i = all_height[start + i]
                    height_raw_ii = all_height[start + ii]
                    if height_raw_ii > height_raw_i:
                        all_height_suppressed[start + i] = 0.
                    elif height_raw_ii < height_raw_i:
                        all_height_suppressed[start + ii] = 0.
    all_height_suppressed = all_height_summed * all_height_suppressed

    return all_height_summed, all_height_suppressed


def load_ms(ws):
    ms = Tims(ws)
    device = ms.get_device_name()
    gradient = ms.get_scan_rts()[-1] / 60.
    logger.info('device_name: {}, gradient: {:.2f}min'.format(device, gradient))
    return ms


class Tims():
    @profile
    def __init__(self, dir_d):
        logger.info('Loading .d data...')
        self.dir_d = dir_d

        self.bruker = bruker.TimsTOF(str(dir_d))
        self.df_settings, self.frames_num_per_cycle = self.get_dia_windows()
        self.im_gap = self.get_im_gap()

        d_ms1_maps, d_ms2_maps = {}, {}
        for swath_id in range(len(self.get_swath())):
            info = 'construct {} map ...'.format(swath_id)
            # print(info)
            map = self.extract_swath_map(swath_id)
            if swath_id == 0:  # ms1
                d_ms1_maps = self.split_ms1_to_chunks(map)
            else:  # ms2
                d_ms2_maps[swath_id] = map

        self.d_ms1_maps = d_ms1_maps
        self.d_ms2_maps = d_ms2_maps

        logger.info('Loading .d data finished.')

    @property
    def frame_nums(self):
        return len(self.bruker.frames)

    def get_dia_windows(self):
        '''
        return:
            df_cycle[
                frame1: [w1, w2, w3, w4]
                frame2: [w1, w2, w3, w4]
            ]
            w: ((im_low, im_high), (q_low, q_high))
        '''
        # MsMsType: 0 -- MS, 9 -- MS/MS
        ms1_idx = np.where(self.bruker.frames.MsMsType == 0)[0]
        ms1_frame_diff = np.diff(ms1_idx)
        assert ms1_frame_diff[0] == 1, 'AlphaTims not add zeroth frame!'
        condition = (ms1_frame_diff[1:] == ms1_frame_diff[1]).all()
        # assert condition, 'alphatims data exists missing cycles!'

        frames_num_per_cycle = ms1_frame_diff[1]  # has a frame for MS1
        df_v = []
        for i in range(2 + 100 * frames_num_per_cycle,
                       100 * frames_num_per_cycle + frames_num_per_cycle +
                       1):
            df = self.bruker[i]
            df = df[df['precursor_indices'] > 0]  # remove overlap ms1 ions
            df = df[['mobility_values',
                     'quad_low_mz_values',
                     'quad_high_mz_values']]
            df_max = df.groupby(['quad_low_mz_values',
                                 'quad_high_mz_values'], sort=False).apply(
                np.maximum.reduce).reset_index(drop=True)
            df_min = df.groupby(['quad_low_mz_values',
                                 'quad_high_mz_values'], sort=False).apply(
                np.minimum.reduce).reset_index(drop=True)
            df = df_max.merge(df_min,
                              on=['quad_low_mz_values',
                                  'quad_high_mz_values'],
                              suffixes=('_max', '_min'))
            df_v.append(df)
        df = pd.concat(df_v).reset_index(drop=True)
        return df, frames_num_per_cycle

    def plot_dia_windows(self):
        fig, ax = plt.subplots()

        x = np.linspace(300, 1300, 50)
        y = np.linspace(0.5, 1.7, 50)

        ax.plot(x, y, 'w')
        df = self.get_dia_windows()[0]
        for i in range(len(df)):
            x_min = df['quad_low_mz_values'][i]
            x_max = df['quad_high_mz_values'][i]
            y_min = df['mobility_values_min'][i]
            y_max = df['mobility_values_max'][i]
            width = x_max - x_min
            height = y_max - y_min
            ax.add_patch(Rectangle((x_min, y_min), width, height,
                                   fc='none',
                                   color='blue'))
        plt.show()

    def get_swath(self):
        '''
        df_settings -- (q_low, q_high) -- ascending
        '''
        x = self.df_settings[['quad_low_mz_values', 'quad_high_mz_values']]
        x = x.drop_duplicates()
        x = x.sort_values(by='quad_low_mz_values')
        low = x['quad_low_mz_values'].values
        high = x['quad_high_mz_values'].values
        # assert (low[1:] == high[0:-1]).all(), 'dia swath exists ' \
        #                                       'overlap between ' \
        #                                       'windows!'
        low[1:] = (low[1:] + high[:-1]) / 2
        swath = np.concatenate([low, [high[-1]]])
        return swath

    def double_sort(self, mzs, ims):
        df = pd.DataFrame({'mz': mzs, 'im': ims})
        idx = df.sort_values(by=['mz', 'im'], ascending=[True, False]).index
        return idx.values

    @profile
    def extract_swath_map(self, window_id):
        '''
        window_id: 0 is MS1
        '''
        all_rt = self.bruker.rt_values
        ms1_idx_v = np.where(self.bruker.frames.MsMsType == 0)[0]
        frame_start = ms1_idx_v[1]
        frame_end = ms1_idx_v[-1]
        all_rt = all_rt[frame_start: frame_end]  # remove start and end
        all_rt = all_rt.astype(np.float32)
        msms_type = self.bruker.frames.MsMsType[frame_start: frame_end]
        ms1_idx_v = np.where(msms_type == 0)[0]

        # frame_len, height, tof
        frame_lens = self.bruker.frames.NumPeaks.values
        all_height = self.bruker.intensity_values # uint16
        all_tof = self.bruker.tof_indices # uint32

        # push to each ion
        push_lens = np.diff(self.bruker.push_indptr)
        assert (len(self.bruker.frames) * self.bruker.scan_max_index ==
                len(push_lens)), 'push exists missing values!'
        push_lens = push_lens.astype(np.uint16)
        push_idx = np.arange(len(push_lens)) % self.bruker.scan_max_index
        push_idx = push_idx.astype(np.int16) # existing subtraction
        all_push = numba_paral_repeat(push_idx, self.bruker.push_indptr)

        # ion -- window
        swath = self.get_swath()
        quad_center_values = self.bruker.quad_mz_values.mean(axis=1)
        quad_window_ids = np.digitize(quad_center_values, swath)
        quad_window_ids = quad_window_ids.astype(np.uint8)

        # by swath_id
        select_id = quad_window_ids == window_id
        select_id = numba_paral_repeat(select_id, self.bruker.quad_indptr)

        frame_len_cumsum = np.cumsum(frame_lens)
        frame_valid_lens = numba_paral_sum(select_id, frame_len_cumsum)

        before_num = frame_lens[:frame_start].sum()
        end_num = frame_lens[frame_end:].sum()
        select_id[0:before_num] = False
        if end_num > 0:
            select_id[-end_num:] = False
        frame_valid_lens = frame_valid_lens[frame_start: frame_end]

        all_push, all_tof, all_height = numba_index_by_bool(
            select_id, all_push, all_tof, all_height
        )

        assert len(all_rt) == len(frame_valid_lens)
        assert (len(all_push) == len(all_tof) ==
                len(all_height) == frame_valid_lens.sum())

        # cycle rt == first frame rt
        all_rt = all_rt[ms1_idx_v]
        cycle_valid_lens = np.add.reduceat(frame_valid_lens, ms1_idx_v)

        # in cycle: mz in ascending order, im not consideration
        cycle_len_cumsum = np.cumsum(cycle_valid_lens)
        result = numba_paral_sort(all_tof, all_push, all_height,
                                  cycle_len_cumsum)
        all_tof, all_push, all_height = result

        # centroid
        tol_push = self.get_centroid_tol_push()
        tol_tof_summed, tol_tof_suppression = 2, 1
        summed2, all_height2 = numba_paral_centroid(
            all_tof,
            all_push,
            all_height,
            tol_tof_summed, tol_tof_suppression, tol_push, cycle_len_cumsum
        )

        tmp = np.split(all_height2, cycle_len_cumsum[:-1])
        cycle_valid_lens2 = np.array(list(map(np.count_nonzero, tmp)))

        select_id = all_height2 > 0
        all_push2, all_tof2, all_height2 = numba_index_by_bool(
            select_id, all_push, all_tof, all_height2
        )
        assert len(all_tof2) == sum(cycle_valid_lens2)

        # push -- im，tof -- m/z
        push_to_im = self.bruker.mobility_values.astype(np.float32)
        all_push = push_to_im[all_push]
        all_push2 = push_to_im[all_push2]

        tof_to_mz = self.bruker.mz_values.astype(np.float32)
        all_tof = tof_to_mz[all_tof]
        all_tof2 = tof_to_mz[all_tof2]

        return (all_rt,
                cycle_valid_lens, all_push, all_tof, all_height,
                cycle_valid_lens2, all_push2, all_tof2, all_height2,
                )

    def get_rt_range(self):
        (
            all_rt, cycle_valid_lens, all_push,
            all_tof, all_height
        ) = self.d_ms1_maps[1]
        return (all_rt.min(), all_rt.max())

    @profile
    def copy_map_to_gpu(self, swath_id, centroid):
        result = []
        for map_type in ['ms1', 'ms2']:
            if map_type == 'ms1':
                (
                    all_rt,
                    cycle_valid_lens, all_push, all_tof, all_height,
                    cycle_valid_lens2, all_push2, all_tof2, all_height2
                ) = self.d_ms1_maps[swath_id]
            else:
                (
                    all_rt,
                    cycle_valid_lens, all_push, all_tof, all_height,
                    cycle_valid_lens2, all_push2, all_tof2, all_height2
                ) = self.d_ms2_maps[swath_id]

            if centroid:
                scan_seek_idx = np.concatenate([
                    [0], np.cumsum(cycle_valid_lens2)], dtype=np.int64
                )
                scan_seek_idx = cuda.to_device(scan_seek_idx)
                scan_im = cuda.to_device(all_push2)
                scan_mz = cuda.to_device(all_tof2)
                scan_height = cuda.to_device(all_height2)
            else:
                scan_seek_idx = np.concatenate(
                    [[0], np.cumsum(cycle_valid_lens)], dtype=np.int64
                )
                scan_seek_idx = cuda.to_device(scan_seek_idx)
                scan_im = cuda.to_device(all_push)
                scan_mz = cuda.to_device(all_tof)
                scan_height = cuda.to_device(all_height)

            dia_map = {
                'scan_rts': all_rt,
                'scan_seek_idx': scan_seek_idx,
                'scan_im': scan_im,
                'scan_mz': scan_mz,
                'scan_height': scan_height
            }
            result.append(dia_map)

        return result

    @profile
    def split_ms1_to_chunks(self, ms1_map):
        '''
        MS1 can split by swath_id to save memory.
        Also, the start and end add 3Da to cover isos of prs.
        '''
        mass_neutron = 1.0033548378
        (
            all_rt,
            cycle_valid_lens, all_push, all_tof, all_height,
            cycle_valid_lens2, all_push2, all_tof2, all_height2
        ) = ms1_map

        # profile and centroid
        scans_seek_idx = np.concatenate([[0], np.cumsum(cycle_valid_lens)])
        scans_seek_idx2 = np.concatenate([[0], np.cumsum(cycle_valid_lens2)])

        swath = self.get_swath()
        d_ms1_maps = {}
        for i in range(len(swath) - 1):
            map_id = i + 1
            pr_mz_low = swath[i] - 3 * mass_neutron
            pr_mz_high = swath[i + 1] + 3 * mass_neutron
            locals_mz, locals_im, locals_height, locals_len = [], [], [], []
            locals_mz2, locals_im2, locals_height2, locals_len2 = [], [], [], []

            for j in range(len(all_rt)):
                # profile
                scan_seek_start = scans_seek_idx[j]
                scan_seek_end = scans_seek_idx[j + 1]
                scan_mz = all_tof[scan_seek_start: scan_seek_end]
                scan_height = all_height[scan_seek_start: scan_seek_end]
                scan_im = all_push[scan_seek_start: scan_seek_end]
                good_idx = (scan_mz >= pr_mz_low) & (scan_mz <= pr_mz_high)
                good_num = good_idx.sum()
                if good_num:
                    local_im, local_mz, local_height = numba_index_by_bool(
                        good_idx, scan_im, scan_mz, scan_height
                    )
                else:
                    local_mz = np.array([10.], dtype=np.float32)
                    local_height = np.array([1], dtype=np.uint16)
                    local_im = np.array([1.], dtype=np.float32)
                locals_mz.append(local_mz)
                locals_im.append(local_im)
                locals_height.append(local_height)
                locals_len.append(len(local_mz))

                # centroid
                scan_seek_start2 = scans_seek_idx2[j]
                scan_seek_end2 = scans_seek_idx2[j + 1]
                scan_mz2 = all_tof2[scan_seek_start2: scan_seek_end2]
                scan_height2 = all_height2[scan_seek_start2: scan_seek_end2]
                scan_im2 = all_push2[scan_seek_start2: scan_seek_end2]
                good_idx = (scan_mz2 >= pr_mz_low) & (scan_mz2 <= pr_mz_high)
                good_num = good_idx.sum()
                if good_num:
                    local_im2, local_mz2, local_height2 = numba_index_by_bool(
                        good_idx, scan_im2, scan_mz2, scan_height2
                    )
                else:
                    local_mz2 = np.array([10.], dtype=np.float32)
                    local_height2 = np.array([1], dtype=np.uint32)
                    local_im2 = np.array([1.], dtype=np.float32)
                locals_mz2.append(local_mz2)
                locals_im2.append(local_im2)
                locals_height2.append(local_height2)
                locals_len2.append(len(local_mz2))

            locals_mz = np.concatenate(locals_mz)
            locals_im = np.concatenate(locals_im)
            locals_height = np.concatenate(locals_height)
            locals_len = np.array(locals_len)

            locals_mz2 = np.concatenate(locals_mz2)
            locals_im2 = np.concatenate(locals_im2)
            locals_height2 = np.concatenate(locals_height2)
            locals_len2 = np.array(locals_len2)

            d_ms1_maps[map_id] = (
                all_rt,
                locals_len, locals_im, locals_mz, locals_height,
                locals_len2, locals_im2, locals_mz2, locals_height2
            )
        return d_ms1_maps

    def get_scan_rts(self):
        scan_rts = self.d_ms2_maps[1][0]
        return scan_rts

    def get_im_gap(self):
        # method-1
        im_min = self.bruker.mobility_min_value
        im_max = self.bruker.mobility_max_value
        im_count = self.bruker.frames.NumScans.max() + 1
        im_gap = (im_max - im_min) / im_count
        return im_gap

        # method-2
        # df = self.bruker[1]
        # x = df['push_indices'].values
        # idx = np.where(x[1:] - x[:-1] == 1)[0]
        # y = df['mobility_values'].values[idx].astype(np.float32)
        # yy = df['mobility_values'].values[idx+1].astype(np.float32)
        # z = y - yy
        # assert z.min() == z.max() # two pushes are not the same on im
        # return z[0]

    def get_centroid_tol_push(self):
        im_range = self.bruker.mobility_max_value - \
                   self.bruker.mobility_min_value
        tol_push = 10 * self.bruker.scan_max_index / 900 / im_range
        return int(tol_push)

    def get_device_name(self):
        _, df1, _, df_NCE, _ = alphatims.bruker.read_bruker_sql(str(self.dir_d))
        d = df1.set_index('Key')['Value'].to_dict()
        return d['InstrumentName']
