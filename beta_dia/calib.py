import warnings

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import MatplotlibDeprecationWarning
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde

from beta_dia import param_g
from beta_dia import utils
from beta_dia.log import Logger

warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

try:
    profile
except NameError:
    profile = lambda x: x

logger = Logger.get_logger()

def update_info_rt(df_seed, df_lib):
    '''
    Calib RT using Calib-RT algorithm.
    Args:
        df_seed: one pr, one best profile group
        df_lib: that need to be updated

    Returns:
        df_lib: after updating
    '''
    idx_max = df_seed.groupby(['locus'])['score_deep'].idxmax()
    df = df_seed.loc[idx_max].reset_index(drop=True)

    idx_max = df.groupby(['pred_irt'])['score_deep'].idxmax()
    df = df.loc[idx_max].reset_index(drop=True)

    idx_max = df.groupby('simple_seq')['score_deep'].idxmax()
    df = df.loc[idx_max].reset_index(drop=True)

    x = df['pred_irt'].values
    y = df['measure_rt'].values
    idx = np.argsort(x)
    x0 = x[idx]
    y0 = y[idx]
    if param_g.is_compare_mode:
        np.savez(param_g.dir_out_single / 'update_rt.npz', x=x, y=y)

    # Calib-RT
    x1, y1, _ = screen_by_hist(x0, y0, bins=100)
    x2, y2 = screen_by_graph(x1, y1)
    x3, y3 = polish_ends(x2, y2, tol_bins=5)

    confident_range = (x0 > x3.min()) & (x0 < x3.max())
    x0, y0 = x0[confident_range], y0[confident_range]

    x_fit, y_fit = fit_by_lowess(x3, y3, frac=0.2)
    f = interp1d(x_fit, y_fit, kind='cubic', fill_value='extrapolate')
    tol = cal_turning_point(y0, f(x0))
    y1_hat = f(x1)
    good_idx = np.abs(y1 - y1_hat) < 1.5 * tol
    x11, y11 = x1[good_idx], y1[good_idx]

    x_fit, y_fit = fit_by_lowess(x11, y11, frac=0.2)
    f = interp1d(x_fit, y_fit, kind='cubic', fill_value='extrapolate')

    # tol is for df_seed, tol_rt is for global extraction
    tol_turn = cal_turning_point(y0, f(x0))
    tol_ratio = param_g.tol_rt
    param_g.tol_rt = max(tol_ratio, tol_turn)
    info = 'tol_rt, by ratio: {:.2f}, by seed: {:.2f}, final: {:.2f}'.format(
        tol_ratio, tol_turn, param_g.tol_rt
    )
    logger.info(info)

    # pred and screen for df_seed
    x_new = df['pred_irt'].values
    y_new = f(x_new).astype(np.float32)
    df['pred_rt'] = y_new
    df['bias_rt'] = df['pred_rt'] - df['measure_rt']
    df = df[df['bias_rt'].abs() < tol]
    df = df[(df['measure_pr_mz'] > 0) & (df['measure_im'] > 0)]
    df = df.reset_index(drop=True)
    bias = df['pred_rt'].values - df['measure_rt'].values

    info = 'After maximum for each locus-time and tol_turn, #seed: {}'.format(
        len(df)
    )
    logger.info(info)
    utils.cal_acc_recall(param_g.ws_single, df, diann_q_pr=0.01)

    # pred for df_lib
    x_new = df_lib['pred_irt'].values
    y_new = f(x_new).astype(np.float32)
    y_new[y_new < 0.] = 0.
    df_lib['pred_rt'] = y_new

    if param_g.is_compare_mode:
        plot_fit_rt(x, y,
                    x1, y1,
                    x11, y11,
                    x_fit, y_fit,
                    tol,
                    bias,
                    fname='update_info_rt')

    cal_rt_recall(param_g.ws_single, df_lib, param_g.tol_rt)

    return df, df_lib


def update_info_im(df_tol, df_lib):
    '''
    Calib IM based on df_tol, and update df_lib
    Args:
        df_tol: from the screen result of df_seed
        df_lib: that need to be updated

    Returns:
        df_lib
    '''
    cal_im_recall(param_g.ws_single, df_lib, param_g.tol_im_xic)
    cal_rt_im_recall(param_g.ws_single, df_lib, param_g.tol_rt, param_g.tol_im_xic)

    idx_max = df_tol.groupby('pred_iim')['score_deep'].idxmax()
    df_tol = df_tol.loc[idx_max].reset_index(drop=True)

    x = df_tol['pred_iim'].values
    y_measure = df_tol['measure_im'].values
    y_pred_before = df_tol['pred_im'].values
    bias_before = y_measure - y_pred_before
    idx = np.argsort(x)
    x, y_pred_before = x[idx], y_pred_before[idx]
    y_measure, bias_before = y_measure[idx], bias_before[idx]

    # lowess
    lowess = sm.nonparametric.lowess
    frac = 0.1
    y_lowess = lowess(y_measure, x, frac=frac)
    x_fit, y_fit = zip(*y_lowess)
    x_fit, y_fit = np.array(x_fit), np.array(y_fit)
    f = interp1d(x_fit, y_fit, kind='cubic', fill_value='extrapolate')

    # 3σ
    y_pred = f(x)
    bias = y_measure - y_pred
    bias_std = (bias - bias.mean()) / bias.std()
    good_idx = bias_std < 3.

    x_good = x[good_idx]
    y_measure_good = y_measure[good_idx]
    y_pred_before = y_pred_before[good_idx]
    bias_before = bias_before[good_idx]
    y_pred_after = f(x_good)
    bias_after = y_measure_good - y_pred_after

    # param_g.tol_im = np.abs(bias_after).max()
    # info = 'updated tol_im: {:.4f}'.format(param_g.tol_im)
    # logger.info(info)
    logger.info('Keep tol_im: 0.05')

    # pred and screen for df_seed
    pred_ims = f(df_tol['pred_iim'].values).astype(np.float32)
    df_tol['pred_im'] = pred_ims

    df_tol['bias_im'] = (df_tol['pred_im'] - df_tol['measure_im']).abs()
    df_tol = df_tol[df_tol['bias_im'] < param_g.tol_im_xic]
    df_tol = df_tol.reset_index(drop=True)

    # pred for df_lib
    pred_ims = f(df_lib['pred_iim'].values).astype(np.float32)
    df_lib['pred_im'] = pred_ims

    if param_g.is_compare_mode:
        plot_fit_im(y_measure_good,
                    y_pred_before, y_pred_after,
                    x_fit, y_fit,
                    bias_before, bias_after,
                    fname='update_info_im')

    cal_im_recall(param_g.ws_single, df_lib, param_g.tol_im_xic)
    cal_rt_im_recall(param_g.ws_single, df_lib, param_g.tol_rt, param_g.tol_im_xic)

    return df_tol, df_lib


@profile
def update_info_mz(df_seed, ms):
    '''
    Calib m/z, update the measure m/z
    Args:
        df_seed:
        ms: that need to be updated

    Returns:
        df_seed
    '''
    idx_max = df_seed.groupby('measure_pr_mz')['score_deep'].idxmax()
    df = df_seed.loc[idx_max].reset_index(drop=True)

    x = df['measure_pr_mz'].values
    y = df['pr_mz'].values
    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]
    bias_old = (x - y) * 1000000. / y

    lowess = sm.nonparametric.lowess
    frac = 0.1
    y_lowess = lowess(y, x, frac=frac)
    x_fit, y_fit = zip(*y_lowess)
    x_fit, y_fit = np.array(x_fit), np.array(y_fit)

    f = interp1d(x_fit, y_fit, kind='cubic', fill_value='extrapolate')

    y_pred = f(x)
    bias = (y - y_pred) * 1000000. / y
    bias_std = (bias - bias.mean()) / bias.std()
    good_idx = np.abs(bias_std) < 3.
    x_good = x[good_idx]
    y_good = y[good_idx]
    y_pred = y_pred[good_idx]
    bias_old = bias_old[good_idx]
    bias_after = bias[good_idx]

    # update tol_ppm
    # param_g.tol_ppm = np.abs(bias_after).max()
    # info = 'updated tol_ppm: {:.2f}'.format(param_g.tol_ppm)
    # logger.info(info)
    logger.info('Keep tol_ppm: 20')

    if param_g.is_compare_mode:
        plot_fit_mz(x_good, y_good,
                    y_pred, y_good,
                    x_fit, y_fit,
                    bias_old, bias_after,
                    fname='update_info_mz')

    # update
    for swath_id in range(len(ms.get_swath())):
        if swath_id == 0:
            continue
        for ms_type in ['ms1', 'ms2']:
            if ms_type == 'ms1':
                ms_map = ms.d_ms1_maps[swath_id]
            else:
                ms_map = ms.d_ms2_maps[swath_id]
            (all_rt,
             cycle_valid_lens, all_push, all_tof, all_height,
             cycle_valid_lens2, all_push2, all_tof2, all_height2
             ) = ms_map

            all_tof = f(all_tof)
            all_tof = all_tof.astype(np.float32)
            all_tof2 = f(all_tof2)
            all_tof2 = all_tof2.astype(np.float32)

            ms_map = (all_rt,
                      cycle_valid_lens, all_push, all_tof, all_height,
                      cycle_valid_lens2, all_push2, all_tof2, all_height2)

            if ms_type == 'ms1':
                ms.d_ms1_maps[swath_id] = ms_map
            else:
                ms.d_ms2_maps[swath_id] = ms_map

    return df_seed


def screen_by_hist(x_data, y_data, bins):
    # find high density points based on hist
    extenti = (x_data.min(), x_data.max())
    extentj = (y_data.min(), y_data.max())
    hist, edges_x, edges_y = np.histogram2d(x_data, y_data, bins=bins,
                                            range=(extenti, extentj))
    edges_x = (edges_x[:-1] + edges_x[1:]) / 2
    edges_y = (edges_y[:-1] + edges_y[1:]) / 2

    cell_idXy = np.stack((np.arange(hist.shape[0]), hist.argmax(axis=1)),
                         axis=-1)  # x -> max(y)
    cell_idYx = np.stack((hist.argmax(axis=0), np.arange(hist.shape[1])),
                         axis=-1)  # y -> max(x)
    cell_idxy = np.vstack((cell_idXy, cell_idYx))
    cell_idxy = np.unique(cell_idxy, axis=0)  # 去重
    cell_idxy = cell_idxy[~(cell_idxy == 0).any(axis=1)]  # remove 0

    x_hist = edges_x[cell_idxy[:, 0]]
    y_hist = edges_y[cell_idxy[:, 1]]
    cell_counts = hist[tuple(cell_idxy.T)]

    return x_hist, y_hist, cell_counts


def screen_by_graph(x_screen1, y_screen1):
    # find the longest length path
    G = nx.DiGraph()
    for i in range(len(x_screen1)):
        x_curr, y_curr = x_screen1[i], y_screen1[i]
        candidates_idx = (x_screen1 >= x_curr) & (y_screen1 >= y_curr)
        candidates_idx[i] = False # no self-cycle
        x_candidates = x_screen1[candidates_idx]
        y_candidates = y_screen1[candidates_idx]
        candidates = [((x_curr, y_curr), (x, y)) for x, y in
                      zip(x_candidates, y_candidates)]
        G.add_edges_from(candidates)

    # maybe exist multiple longest paths
    longest_path = nx.dag_longest_path(G)
    x_screen2, y_screen2 = zip(*longest_path)
    x_screen2, y_screen2 = np.array(x_screen2), np.array(y_screen2)

    return x_screen2, y_screen2


def polish_ends(x_screen2, y_screen2, tol_bins):
    # left
    center_idx = int(len(x_screen2) / 2)
    x, y = x_screen2[:center_idx], y_screen2[:center_idx]
    stepx = x[1:] - x[:-1]
    good_x = (stepx / stepx[stepx > 0].min()) < tol_bins
    stepy = y[1:] - y[:-1]
    good_y = (stepy / stepy[stepy > 0].min()) < tol_bins
    good_xy = good_x & good_y
    breaks_idx = np.where(good_xy == False)[0]
    break_idx = 0
    if len(breaks_idx) > 0:
        idx = np.where(breaks_idx < len(x) * 0.25)[0]
        if len(idx) > 0:
            break_idx = breaks_idx[idx][-1] + 1
    x_left, y_left = x[break_idx:], y[break_idx:]

    # right
    x, y = x_screen2[center_idx:], y_screen2[center_idx:]
    stepx = x[1:] - x[:-1]
    good_x = (stepx / stepx[stepx > 0].min()) < tol_bins
    stepy = y[1:] - y[:-1]
    good_y = (stepy / stepy[stepy > 0].min()) < tol_bins
    good_xy = good_x & good_y
    breaks_idx = np.where(good_xy == False)[0]
    break_idx = len(x)
    if len(breaks_idx) > 0:
        idx = np.where(breaks_idx > len(x) * 0.75)[0]
        if len(idx) > 0:
            break_idx = breaks_idx[idx][0] + 1
    x_right, y_right = x[:break_idx], y[:break_idx]

    x = np.concatenate([x_left, x_right])
    y = np.concatenate([y_left, y_right])
    return x, y


def fit_by_lowess(x, y, frac):
    lowess = sm.nonparametric.lowess
    y_lowess = lowess(y, x, frac=frac)
    x_fit, y_fit = zip(*y_lowess)
    x_fit, y_fit = np.array(x_fit), np.array(y_fit)

    # drop duplicates
    _, idx = np.unique(x_fit, return_index=True)
    x_fit, y_fit = x_fit[idx], y_fit[idx]

    return x_fit, y_fit


def cal_turning_point(y_data, y_pred):
    tol_rt_v = np.arange(1., int(0.5 * y_data.max()), 0.1)
    cover_num_v = []
    for tol_rt in tol_rt_v:
        bias = y_pred - y_data
        cover_num = np.sum(np.abs(bias) < tol_rt)
        cover_num_v.append(cover_num)
    cover_nums = np.array(cover_num_v)

    curve = cover_nums
    nPoints = len(curve)
    allCoord = np.vstack((tol_rt_v, curve)).T
    np.array([range(nPoints), curve])
    firstPoint = allCoord[0]
    lineVec = allCoord[-1] - allCoord[0]
    lineVecNorm = lineVec / np.sqrt(np.sum(lineVec ** 2))
    vecFromFirst = allCoord - firstPoint
    scalarProduct = np.sum(
        vecFromFirst * np.tile(lineVecNorm, (nPoints, 1)), axis=1)
    vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)
    vecToLine = vecFromFirst - vecFromFirstParallel
    distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))
    best_idx = np.argmax(distToLine)
    tol_rt = tol_rt_v[best_idx]

    return tol_rt


def plot_fit_rt(x, y, x1, y1, x11, y11, x_fit, y_fit,
                tol_rt,
                bias,
                fname):
    plt.clf()
    fig, ax = plt.subplots(2, 1)
    # raw points
    ax[0].plot(x, y, 'bo', label='Raw', markersize=1.)
    # high density points
    ax[0].plot(x1, y1, 'go', label='Hist-Global', markersize=1.)
    # path points
    ax[0].plot(x11, y11, 'ro', label='Hist-Local', markersize=1.)
    # fit
    ax[0].plot(x_fit, y_fit, label='Fit')
    ax[0].plot(x_fit, y_fit + tol_rt, label='Fit+')
    ax[0].plot(x_fit, y_fit - tol_rt, label='Fit-')
    # bias
    ax[1].hist(bias, bins=20, label='Bias of seeds')
    ax[1].axvline(x=tol_rt)

    for a in ax:
        a.legend()
    plt.tight_layout()
    plt.savefig(param_g.dir_out_single / (fname + '.png'), bbox_inches='tight')


def plot_fit_im(y_measure, y_pred_before, y_pred_after,
                xout, yout,
                bias_old, bias, fname):
    plt.clf()
    fig, ax = plt.subplots(5, 1)

    # before
    x = y_pred_before
    y = y_measure - x
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    xlim = (x.min(), x.max())
    ax[0].scatter(x, y, s=1, c=z, label='Data points')
    ax[0].set_xlim(xlim)
    ax[0].set_ylim((-0.05, 0.05))

    # after
    x = y_pred_after
    y = y_measure - x
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    ax[1].scatter(x, y, s=1, c=z, label='Good Points')
    ax[1].set_xlim(xlim)
    ax[1].set_ylim((-0.05, 0.05))

    # Deviation trend line
    ax[2].plot(xout, (yout - xout), markersize=1.)

    # bias before
    xlim = (-0.05, 0.05)
    ax[3].hist(bias_old, bins=20)
    ax[3].set_xlim(xlim)

    # bias after
    ax[4].hist(bias, bins=20)
    ax[4].set_xlim(xlim)

    # plt.legend()
    plt.tight_layout()
    plt.savefig(param_g.dir_out_single / (fname + '.png'), bbox_inches='tight')


def plot_fit_mz(x1, y1, x2, y2, x_fit, y_fit, bias_old, bias, fname):
    plt.clf()
    fig, ax = plt.subplots(5, 1)

    # data before
    x = x1
    y = (y1 - x1) * 1e6 / y1
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    xlim = (x.min(), x.max())
    ax[0].scatter(x, y, s=1, c=z, label='Data points')
    ax[0].set_xlim(xlim)
    ax[0].set_ylim((-20, 20))

    # data after
    x = x2
    y = (y2 - x2) * 1e6 / y1
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    ax[1].scatter(x, y, s=1, c=z, label='Data points')
    ax[1].set_xlim(xlim)
    ax[1].set_xlim(xlim)
    ax[1].set_ylim((-20, 20))

    # trend line
    ax[2].plot(x_fit, (y_fit - x_fit) * 1e6 / y_fit, label='Fit line')

    xlim = (bias_old.min(), bias_old.max())
    ax[3].hist(bias_old, bins=20)
    ax[3].set_xlim(xlim)
    ax[4].hist(bias, bins=20)
    ax[4].set_xlim(xlim)

    # plt.legend()
    plt.tight_layout()
    plt.savefig(param_g.dir_out_single / (fname + '.png'), bbox_inches='tight')


def cal_rt_recall(ws, df_lib, tol_rt):
    if not param_g.is_compare_mode:
        return
    df_diann = pd.read_csv(ws / 'diann' / 'report.tsv', sep='\t')
    df_diann = df_diann[df_diann['Q.Value'] < 0.01]
    df_diann['pr_id'] = (df_diann['Modified.Sequence'] +
                         df_diann['Precursor.Charge'].astype(str))
    df_diann['RT'] = df_diann['RT'] * 60.

    df_diann = df_diann[['pr_id', 'RT']]
    if 'group_rank' in df_lib.columns:
        df = df_diann.merge(df_lib[df_lib.group_rank == 1], on='pr_id',
                            how='left')
    else:
        df = df_diann.merge(df_lib, on='pr_id', how='left')
    df['bias'] = (df['pred_rt'] - df['RT']).abs()
    good_num = (df['bias'][~df['bias'].isna()] < tol_rt).sum()
    recall = good_num / len(df_diann)
    info = 'Based on the pred_rt, recall is: {:.3f}'.format(recall)
    logger.info(info)


def cal_im_recall(ws, df_lib, tol_im):
    if not param_g.is_compare_mode:
        return
    df_diann = pd.read_csv(ws / 'diann' / 'report.tsv', sep='\t')
    df_diann = df_diann[df_diann['Q.Value'] < 0.01]
    df_diann['pr_id'] = (df_diann['Modified.Sequence'] +
                         df_diann['Precursor.Charge'].astype(str))
    df_diann = df_diann[['pr_id', 'IM']]
    if 'group_rank' not in df_lib.columns:
        df = df_diann.merge(df_lib, on='pr_id', how='left')
    else:
        df = df_diann.merge(df_lib[df_lib.group_rank == 1], on='pr_id',
                            how='left')
    df['bias'] = (df['pred_im'] - df['IM']).abs()
    good_num = (df['bias'][~df['bias'].isna()] < tol_im).sum()
    recall = good_num / len(df_diann)
    info = 'Based on the pred_im, recall is: {:.3f}'.format(recall)
    logger.info(info)


def cal_rt_im_recall(ws, df_lib, tol_rt, tol_im):
    if not param_g.is_compare_mode:
        return
    df_diann = pd.read_csv(ws / 'diann' / 'report.tsv', sep='\t')
    df_diann = df_diann[df_diann['Q.Value'] < 0.01]
    df_diann['pr_id'] = (df_diann['Modified.Sequence'] +
                         df_diann['Precursor.Charge'].astype(str))
    df_diann = df_diann[['pr_id', 'RT', 'IM']]
    if 'diann_rt' not in df_lib.columns:
        df_diann['diann_rt'] = df_diann['RT'] * 60.

    if 'group_rank' not in df_lib.columns:
        df = df_diann.merge(df_lib, on='pr_id', how='left')
    else:
        df = df_diann.merge(df_lib[df_lib.group_rank == 1], on='pr_id',
                            how='left')

    df['bias_rt'] = (df['pred_rt'] - df['diann_rt']).abs()
    df['bias_im'] = (df['pred_im'] - df['IM']).abs()

    df = df.dropna(subset=['bias_rt', 'bias_im']).reset_index(drop=True)

    condition1 = (df['bias_rt'] < tol_rt)
    condition2 = (df['bias_im'] < tol_im)
    good_num = sum(condition1 & condition2)
    recall = good_num / len(df_diann)
    info = 'Based on the pred_rt and pred_im, recall is: {:.3f}'.format(recall)
    logger.info(info)
