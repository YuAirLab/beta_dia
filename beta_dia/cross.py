import copy
import numpy as np
import pandas as pd
from functools import reduce

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from beta_dia import param_g
from beta_dia import utils
from beta_dia.log import Logger
from beta_dia import fdr
from beta_dia import assemble
from beta_dia.models import DeepQuant

try:
    profile
except NameError:
    profile = lambda x: x

logger = Logger.get_logger()

def drop_batches_mismatch(df):
    # remove decoy duplicates
    df_decoy = df[df['decoy'] == 1]
    idx_max = df_decoy.groupby('pr_id')['cscore_pr_run'].idxmax()
    df_decoy = df_decoy.loc[idx_max].reset_index(drop=True)

    # remove decoy mismatch
    df_target = df[df['decoy'] == 0]
    bad_idx = df_decoy['pr_id'].isin(df_target['pr_id'])
    df_decoy = df_decoy.loc[~bad_idx]

    df = pd.concat([df_target, df_decoy], axis=0, ignore_index=True)
    assert len(df) == df['pr_id'].nunique()
    return df


def drop_runs_mismatch(df):
    # remove decoy duplicates
    df_decoy = df[df['decoy'] == 1]
    idx_max = df_decoy.groupby('pr_id')['cscore_pr_global'].idxmax()
    df_decoy = df_decoy.loc[idx_max].reset_index(drop=True)

    # remove target duplicates
    df_target = df[df['decoy'] == 0]
    idx_max = df_target.groupby('pr_id')['cscore_pr_global'].idxmax()
    df_target = df_target.loc[idx_max].reset_index(drop=True)

    # remove decoy mismatch
    bad_idx = df_decoy['pr_id'].isin(df_target['pr_id'])
    df_decoy = df_decoy.loc[~bad_idx]

    df = pd.concat([df_target, df_decoy], axis=0, ignore_index=True)
    assert len(df) == df['pr_id'].nunique()
    return df


def cal_global_first(multi_ws, lib, top_k_fg):
    '''
    Generate df_global: a row is a pr with other cross info
    Returns:
        [pr_id, decoy, cscore_pr_run_x]
        [cscore_pr_global_first, q_pr_global_first]
        [proteotypic, protein_id, protein_name, protein_group]
        [cscore_pg_global_first, q_pg_global_first]
        [quant_pr_0, quant_pr_1, ..., quant_pr_N]
    '''
    cols_basic = ['pr_index', 'pr_id', 'decoy', 'cscore_pr_run', 'is_main']
    logger.info(f'Merge {len(multi_ws)} .parquet files ...')
    for ws_i, ws_single in enumerate(multi_ws):
        df_raw = utils.read_from_pq(ws_single, cols_basic)
        df = df_raw[df_raw['is_main']] # main for global, non-main for reanalysis
        del df['is_main']
        if ws_i == 0:
            df_global = df
            df_global = df_global.rename(columns={'cscore_pr_run': 'cscore_pr_global'})
        else:
            df_global = df_global.merge(
                df, on=['pr_id', 'decoy', 'pr_index'], how='outer'
            )
            df_global['cscore_pr_global'] = np.fmax(
                df_global['cscore_pr_run'], df_global['cscore_pr_global']
            )
            del df_global['cscore_pr_run']
    assert df_global.isna().sum().sum() == 0

    # polish prs
    df_global = drop_runs_mismatch(df_global)
    df_global['pr_IL'] = df_global['pr_id'].replace(['I', 'L'], ['x', 'x'], regex=True)
    idx_max = df_global.groupby('pr_IL')['cscore_pr_global'].idxmax()
    df_global = df_global.loc[idx_max].reset_index(drop=True)
    del df_global['pr_IL']

    # q_pr_global
    df_global = fdr.cal_q_pr_core(df_global, run_or_global='global')

    # remove global rubbish
    df_global = df_global[df_global['q_pr_global'] < param_g.rubbish_q_cut]
    df_global = df_global.reset_index(drop=True)
    global_prs = set(df_global['pr_id'])
    utils.print_ids(df_global, 0.05, pr_or_pg='pr', run_or_global='global')

    # load quant info
    cols_quant = ['score_ion_quant_' + str(i) for i in range(0, param_g.fg_num+2)]
    cols_sa = ['score_ion_sa_' + str(i) for i in range(0, param_g.fg_num+2)]
    for ws_i, ws_single in enumerate(multi_ws):
        df_raw = utils.read_from_pq(ws_single, cols_basic + cols_quant + cols_sa)
        df = df_raw[df_raw['is_main']]
        df = df.drop(columns=['is_main', 'cscore_pr_run'])
        df = df[df['pr_id'].isin(global_prs)]
        cols_quant_long = ['run_' + str(ws_i) + '_' + x for x in cols_quant]
        df = df.rename(columns=dict(zip(cols_quant, cols_quant_long)))
        cols_sa_long = ['run_' + str(ws_i) + '_' + x for x in cols_sa]
        df = df.rename(columns=dict(zip(cols_sa, cols_sa_long)))
        df_global = df_global.merge(df, on=['pr_id', 'decoy', 'pr_index'], how='left')

    # assemble: proteotypic, protein_id, protein_name, protein_group
    # cscore_pg_global, q_pg_global
    df_global = lib.assign_proteins(df_global)
    df_global = assemble.assemble_to_pg(df_global, param_g.q_cut_infer, 'global')
    df_global = fdr.cal_q_pg(df_global, param_g.q_cut_infer, 'global')
    utils.print_ids(df_global, 0.05, pr_or_pg='pg', run_or_global='global')

    # cross quant
    df_global = quant_pr_autoencoder(df_global, top_k_fg)
    # df_global = quant_pr_cross(df_global, top_k_fg)

    # return
    df_global = df_global.drop(columns=['pr_index', 'simple_seq'])
    df_global = df_global.loc[:, ~df_global.columns.str.startswith('run_')]
    df_global = df_global.loc[:, ~df_global.columns.str.startswith('cscore_pr_run_')]
    df_global = df_global.rename(columns={
        'cscore_pr_global': 'cscore_pr_global_first',
        'q_pr_global': 'q_pr_global_first',
        'cscore_pg_global': 'cscore_pg_global_first',
        'q_pg_global': 'q_pg_global_first'
    })
    return df_global


def cal_global_update(df_global, bad_seqs):
    # Remove interfered prs from df_global and recalculate cscore and q value
    df_global = df_global[~df_global['pr_id'].isin(bad_seqs)]
    df_global = df_global.reset_index(drop=True)

    df_global['cscore_pr_global'] = df_global['cscore_pr_global_first']
    df_global = fdr.cal_q_pr_core(df_global, run_or_global='global')
    df_global = fdr.cal_q_pg(df_global, param_g.q_cut_infer, 'global')

    utils.print_ids(df_global, 0.05, pr_or_pg='pr', run_or_global='global')
    utils.print_ids(df_global, 0.05, pr_or_pg='pg', run_or_global='global')

    # result
    df_global = df_global.rename(columns={
        'cscore_pr_global': 'cscore_pr_global_second',
        'q_pr_global': 'q_pr_global_second',
        'cscore_pg_global': 'cscore_pg_global_second',
        'q_pg_global': 'q_pg_global_second'
    })
    return df_global


def cal_kde(labels, choice_size=10000):
    from sklearn.model_selection import GridSearchCV
    from sklearn.neighbors import KernelDensity

    # sample labels
    choice_size = min(choice_size, len(labels))
    labels_sample = np.random.choice(labels, size=choice_size, replace=False)

    # init bandwidth by Silverman
    b = 1.06 * np.std(labels_sample) * choice_size**(-1/5)
    bandwidths = np.linspace(0.1 * b, 1.5 * b, 20)

    # grid search for bandwidth
    grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                        param_grid={'bandwidth': bandwidths},
                        cv=5,
                        n_jobs=1 if __debug__ else 8)
    grid.fit(labels_sample[:, None])
    best_bandwidth = grid.best_params_['bandwidth']

    # fit by sample
    best_kde = KernelDensity(kernel='gaussian', bandwidth=best_bandwidth)
    best_kde.fit(labels_sample[:, None])

    # pred for grid
    x_grid = np.linspace(labels.min(), labels.max(), 1000)
    log_density_grid = best_kde.score_samples(x_grid[:, None])
    density_grid = np.exp(log_density_grid)

    # interp
    from scipy.interpolate import interp1d
    interp_func = interp1d(x_grid, density_grid, fill_value="extrapolate")

    # pred for all
    density = interp_func(labels)

    # cal weights
    weights = 1 / (density + 1e-6)
    # weights = np.log2(1 + np.sqrt(weights))
    # weights = (weights - weights.min()) / (weights.max() - weights.min())
    weights = np.clip(weights, None, 200* weights.min())
    return weights


def cal_ratio_sum(X):
    u = np.nanmean(X, axis=1)
    sigma = np.nanstd(X, axis=1)
    bias = np.abs(X - u[:, None]) / sigma[:, None]
    X[bias > 3] = np.nan
    base = np.nanmin(X, axis=1)[:, None]
    ratios = X / base
    ratios = np.log2(ratios)
    ratios_sum = np.nansum(ratios, axis=1)
    return ratios_sum


def quant_pr_autoencoder(df_global, top_k_fg):
    # decoys not in considering
    df_target = df_global[df_global['decoy'] == 0].reset_index(drop=True)
    df_decoy = df_global[df_global['decoy'] == 1].reset_index(drop=True)

    import re
    n_run = max(int(m.group(1)) for col in df_global.columns if
            (m := re.search(r'run_(\d+)', col))
    ) + 1

    sa_m_v, area_m_v, pr_quant_m = [], [], []
    ion_idx = range(2, 2 + param_g.fg_num) # only considering MS2 signal
    for wi in range(n_run):
        cols_sa = ['run_' + str(wi) + '_' + 'score_ion_sa_' + str(i) for i in ion_idx]
        sa_m = df_target.loc[:, cols_sa].values
        sa_m[np.isnan(sa_m)] = 0.
        sa_m_v.append(sa_m)

        cols_quant = ['run_' + str(wi) + '_' + 'score_ion_quant_' + str(i) for i in ion_idx]
        area_m = df_target.loc[:, cols_quant].values

        # pr_quant_v is used for checking ratios distribution,
        # whose zeros equal to na
        pr_quant_v = np.nansum(area_m[:, :top_k_fg], axis=1)
        pr_quant_v[pr_quant_v == 0] = np.nan
        pr_quant_m.append(pr_quant_v)

        # raw values without any replacements
        area_m_v.append(area_m)

    pr_quant_m = np.array(pr_quant_m).T
    ratios = cal_ratio_sum(pr_quant_m)
    weights = cal_kde(ratios)

    # prepare dataset: norm globally or row-wise for area_m_log
    sa_m = np.hstack(sa_m_v)  # [n_pep, n_run * n_ion]
    n_pep, n_ion = len(area_m), area_m_v[0].shape[-1]

    area_m = np.hstack(area_m_v) # [n_pep, n_run * n_ion]
    area_m[np.isnan(area_m) | (area_m < 1.1)] = 1.1
    area_m_log = np.log2(area_m)

    area_m_max1 = area_m_log.max()
    area_m_norm1 = area_m_log / area_m_max1
    area_m_max2 = area_m_log.max(axis=1)
    area_m_norm2 = area_m_log / area_m_max2[:, None]

    # pytorch
    X_area1 = torch.tensor(area_m_norm1).to(param_g.gpu_id)
    X_area2 = torch.tensor(area_m_norm2).to(param_g.gpu_id)
    X_sa = torch.tensor(sa_m).to(param_g.gpu_id)
    W = torch.tensor(weights).to(param_g.gpu_id)

    train_val_idx = df_target['q_pr_global'] < 0.01
    X_area1_train_val = X_area1[train_val_idx]
    X_area2_train_val = X_area2[train_val_idx]
    X_sa_train_val = X_sa[train_val_idx]
    W_train_val = W[train_val_idx]
    dataset = TensorDataset(
        X_area1_train_val, X_area2_train_val, X_sa_train_val, W_train_val
    )
    dataset_pred = TensorDataset(X_area1, X_area2, X_sa)

    train_num = int(0.8 * len(dataset))
    eval_num = len(dataset) - train_num
    logger.info(f'DeepQuant train: {train_num} prs, eval: {eval_num} prs')

    train_set, val_set = torch.utils.data.random_split(
        dataset,
        [train_num, eval_num],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=256)
    pred_loader = DataLoader(dataset_pred, batch_size=1024, shuffle=False)

    model = DeepQuant(n_run, n_ion).to(param_g.gpu_id)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss(reduction='none')

    # train and valid
    best_loss = float('inf')
    epochs_no_improve = 0
    for epoch in range(100):
        # train
        model.train()
        epoch_train_loss = 0
        for X_area1_batch, X_area2_batch, X_sa_batch, W_batch in train_loader:
            optimizer.zero_grad()
            recon = model(X_area1_batch, X_area2_batch, X_sa_batch)
            loss = criterion(recon, X_area1_batch)
            # loss = loss.mean()
            loss = (loss.mean(dim=1) * W_batch).mean()
            epoch_train_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_loss = epoch_train_loss / len(train_loader)

        # val
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for X_area1_val, X_area2_val, X_sa_val, W_batch in val_loader:
                recon_val = model(X_area1_val, X_area2_val, X_sa_val)
                loss = criterion(recon_val, X_area1_val)
                # loss = loss.mean()
                loss = (loss.mean(dim=1) * W_batch).mean()
                epoch_val_loss += loss.item()
        val_loss = epoch_val_loss / len(val_loader)

        if epoch == 0:
            info = 'DeepQuant train epoch: {}, train loss: {:.3f}, eval loss: {:.3f}'.format(
                epoch, train_loss, val_loss
            )
            logger.info(info)

        # stop check
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
            model_best = copy.deepcopy(model)
            info = 'DeepQuant train epoch: {}, train loss: {:.3f}, eval loss: {:.3f}'.format(
                epoch, train_loss, val_loss
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= 10:
                break
    logger.info(info)

    # pred all
    pred_v = []
    model.eval()
    with torch.no_grad():
        for X_area1_batch, X_area2_batch, X_sa_batch in pred_loader:
            X_pred = model_best(X_area1_batch, X_area2_batch, X_sa_batch)
            X_pred = X_pred.cpu().numpy()
            pred_v.append(X_pred)
    pred_m = np.vstack(pred_v)
    pred_m = pred_m * area_m_max1
    pred_m = np.exp2(pred_m)

    # quant
    sa_sum = np.nansum(sa_m_v, axis=0)
    top_n_idx = np.argsort(sa_sum, axis=1)[:, -top_k_fg:]
    for run_idx, area_m in enumerate(area_m_v):
        # raw quant
        top_n_values = np.take_along_axis(area_m, top_n_idx, axis=1)
        pr_quant_raw = top_n_values.sum(axis=1)
        pr_quant_raw[np.isnan(pr_quant_raw)] = 0
        df_target['quant_pr_raw_' + str(run_idx)] = np.float32(0)
        df_target['quant_pr_raw_' + str(run_idx)] = pr_quant_raw

        # deep quant
        area_m_ae = pred_m[:, run_idx*n_ion : (run_idx+1)*n_ion]
        top_n_values = np.take_along_axis(area_m_ae, top_n_idx, axis=1)
        pr_quant_deep = top_n_values.sum(axis=1)
        df_target['quant_pr_deep_' + str(run_idx)] = np.float32(0)
        df_target['quant_pr_deep_' + str(run_idx)] = pr_quant_deep

        # final quant
        df_target['quant_pr_' + str(run_idx)] = np.sqrt(pr_quant_raw * pr_quant_deep)

    df_global = pd.concat([df_target, df_decoy], axis=0, ignore_index=True)
    return df_global
