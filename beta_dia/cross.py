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
from beta_dia import polish

try:
    # profile
    profile = lambda x: x
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


def group_by_species(species, ratio_cut=0.05):
    df = species.value_counts(normalize=True).reset_index() # sorted
    df = df.rename(columns={'protein_name': 'species'})
    mask = df['proportion'] < ratio_cut
    to_merge = df[mask]
    remaining_df = df[~mask]

    if not to_merge.empty:
        merged_a = to_merge['species'].tolist()
        merged_b = to_merge['proportion'].sum()

        while merged_b < ratio_cut and not remaining_df.empty:
            min_b_row = remaining_df.nsmallest(1, 'proportion')
            merged_a += min_b_row['species'].tolist()
            merged_b += min_b_row['proportion'].sum()
            remaining_df = remaining_df.drop(min_b_row.index)

        if merged_b > ratio_cut:
            new_row = pd.DataFrame({'species': [merged_a], 'proportion': [merged_b]})
            df = pd.concat([remaining_df, new_row], ignore_index=True)
        else:
            df = pd.concat([df[~mask], to_merge], ignore_index=True)
    df['species'] = df['species'].apply(lambda x: x if isinstance(x, list) else [x])
    return df['species'].tolist()


def cal_global(lib, top_k_fg, top_k_pr, multi_ws=[], df_v=[], df_global1=None):
    '''
    Generate df_global: a row is a pr with other cross info
    Returns:
        [pr_id, decoy, cscore_pr_run_x]
        [cscore_pr_global_first, q_pr_global_first]
        [proteotypic, protein_id, protein_name, protein_group]
        [cscore_pg_global_first, q_pg_global_first]
        [quant_pr_0, quant_pr_1, ..., quant_pr_N]
        [quant_pg_0, quant_pg_1, ..., quant_pg_N]
    '''
    n_run = max(len(multi_ws), len(df_v))
    logger.info(f'Merge {n_run} .parquet files ...')

    cols_basic = ['pr_index', 'pr_id', 'decoy', 'cscore_pr_run']
    for ws_i in range(n_run):
        if len(multi_ws) > 0:
            df = utils.read_from_pq(multi_ws[ws_i], cols_basic)
            df = df[df['cscore_pr_run'] <= 1.] # only here with filter
        if len(df_v) > 0:
            df = df_v[ws_i][cols_basic]
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
    utils.print_ids(df_global, 0.05, pr_or_pg='pr', run_or_global='global')

    # assemble: proteotypic, protein_id, protein_name, protein_group
    # cscore_pg_global, q_pg_global
    df_global['simple_seq'] = df_global['pr_id'].str[:-1].replace(
        ['C\(UniMod:4\)', 'M\(UniMod:35\)'], ['c', 'm'], regex=True
    )
    df_global['strip_seq'] = df_global['simple_seq'].str.upper()
    if df_global1 is None:
        df_global = lib.assign_proteins(df_global)
        q_cut_v = np.arange(0.01, 0.06, 0.01)
        ids_001_v = []
        for q_cut in q_cut_v:
            df_tmp = df_global[df_global['q_pr_global'] < q_cut]
            df_tmp = df_tmp.reset_index(drop=True).copy()
            df_tmp = assemble.assemble_to_pg(df_tmp, q_cut, 'global')
            df_tmp = fdr.cal_q_pg(df_tmp, q_cut, 'global')
            ids_001 = df_tmp[(df_tmp['q_pg_global'] < 0.01) & (df_tmp['decoy'] == 0)]['protein_group'].nunique()
            ids_001_v.append(ids_001)
        q_cut = q_cut_v[np.argmax(ids_001_v)]
        logger.info(f'Select q_cut: {q_cut:.2f} for pg inference and score')
        param_g.q_cut_infer = q_cut

        df_global = df_global[df_global['q_pr_global'] < q_cut].reset_index(drop=True)
        df_global2 = df_global.copy()
        df_global2['protein_id'] = df_global2['protein_name']

        df_global = assemble.assemble_to_pg(df_global, q_cut, 'global')
        df_global2 = assemble.assemble_to_pg(df_global2, q_cut, 'global')
        x = df_global['protein_group'].str.count(';')
        x2 = df_global2['protein_group'].str.count(';')
        assert (x == x2).all()
        df_global['protein_name'] = df_global2['protein_group']
    else:
        cols = ['pr_id', 'decoy', 'proteotypic', 'protein_id', 'protein_name', 'protein_group']
        df_global = pd.merge(df_global, df_global1[cols], on=['pr_id', 'decoy'])
    df_global = fdr.cal_q_pg(df_global, q_cut, 'global')
    utils.print_ids(df_global, 0.05, pr_or_pg='pg', run_or_global='global')

    # load fg_mz for main eat other
    cols_fg_mz = ['fg_mz_' + str(i) for i in range(param_g.fg_num)]
    df_global = lib.assign_fg_mz(df_global)

    # load quant info
    cols_run = ['swath_id', 'locus', 'measure_im']
    cols_sa = ['score_ion_sa_' + str(i) for i in range(0, param_g.fg_num + 2)]
    cols_quant = ['score_ion_quant_' + str(i) for i in range(0, param_g.fg_num + 2)]
    for ws_i in range(n_run):
        if len(multi_ws) > 0:
            df = utils.read_from_pq(
                multi_ws[ws_i], cols_basic + cols_quant + cols_sa + cols_run
            )
        else:
            df = df_v[ws_i][cols_basic + cols_quant + cols_sa]
        df_global = df_global.merge(df, on=['pr_id', 'decoy', 'pr_index'], how='left')
        df_global = polish.zero_interference_areas(df_global)
        df_global = df_global.drop(cols_run + ['cscore_pr_run'], axis=1)

        cols_quant_long = ['run_' + str(ws_i) + '_' + x for x in cols_quant]
        df_global = df_global.rename(columns=dict(zip(cols_quant, cols_quant_long)))
        cols_sa_long = ['run_' + str(ws_i) + '_' + x for x in cols_sa]
        df_global = df_global.rename(columns=dict(zip(cols_sa, cols_sa_long)))
    df_global = df_global.drop(cols_fg_mz, axis=1)

    # cross quant pr by species
    species = df_global['protein_name'].str.split('_').str[-1]
    df_global['species'] = species
    species = df_global.loc[df_global['q_pr_global'] < 0.01, 'species']
    assert df_global['species'].isin(set(species)).all()
    species_v = group_by_species(species)
    df_tmp_v = []
    for species in species_v:
        logger.info(f'Training DeepQuant on {species} level...')
        df_tmp = df_global[df_global['species'].isin(species)]
        df_tmp = df_tmp.reset_index(drop=True)
        df_tmp_v.append(quant_pr_autoencoder(df_tmp, top_k_fg))
    df_global = pd.concat(df_tmp_v, ignore_index=True)

    del df_global['species']
    df_global = df_global.loc[:, ~df_global.columns.str.startswith('run_')]
    df_global = df_global.loc[:, ~df_global.columns.str.startswith('cscore_pr_run_')]

    # quant pg: select top-k-pr by their q values
    cols_pr_raw = ['quant_pr_raw_' + str(i) for i in range(n_run)]
    cols_pr_deep = ['quant_pr_deep_' + str(i) for i in range(n_run)]
    cols_pr_mix = ['quant_pr_mix_' + str(i) for i in range(n_run)]

    cols_pg_raw = ['quant_pg_raw_' + str(i) for i in range(n_run)]
    cols_pg_deep = ['quant_pg_deep_' + str(i) for i in range(n_run)]
    cols_pg_mix = ['quant_pg_mix_' + str(i) for i in range(n_run)]

    df = df_global.sort_values(by='cscore_pr_global', ascending=False)
    df_top = df.groupby('protein_group').head(top_k_pr)
    df_top = df_top.groupby('protein_group')

    for cols_pr, cols_pg in zip([cols_pr_raw, cols_pr_deep, cols_pr_mix],
                                [cols_pg_raw, cols_pg_deep, cols_pg_mix]):
        df = df_top[cols_pr].mean().reset_index()
        df = df.rename(columns=dict(zip(cols_pr, cols_pg)))
        df_global = pd.merge(df_global, df, on='protein_group', how='left')

    return df_global


def quant_pr_autoencoder(df_global, top_k_fg, is_log_correct=False):
    import re
    n_run = max(int(m.group(1)) for col in df_global.columns if
            (m := re.search(r'run_(\d+)', col))
    ) + 1

    sa_m_v, area_m_v, pr_quant_m = [], [], []
    ion_idx = range(2, 2 + param_g.fg_num) # only considering MS2 signal
    for wi in range(n_run):
        cols_sa = ['run_' + str(wi) + '_' + 'score_ion_sa_' + str(i) for i in ion_idx]
        sa_m = df_global.loc[:, cols_sa].values
        sa_m[np.isnan(sa_m)] = 0.
        sa_m_v.append(sa_m)

        cols_quant = ['run_' + str(wi) + '_' + 'score_ion_quant_' + str(i) for i in ion_idx]
        area_m = df_global.loc[:, cols_quant].values
        area_m[np.isnan(area_m) | (area_m < 1.1)] = 1.1 # maybe transformed by log
        area_m_v.append(area_m)

    # prepare dataset: norm globally or row-wise for area_m_log
    sa_m = np.hstack(sa_m_v)  # [n_pep, n_run * n_ion]
    n_pep, n_ion = len(area_m), area_m_v[0].shape[-1]

    area_m = np.hstack(area_m_v) # [n_pep, n_run * n_ion]
    area_m_log = np.log2(area_m)

    area_m_log_max1 = area_m_log.max()
    area_m_norm1 = area_m_log / area_m_log_max1
    area_m_norm_min1 = area_m_norm1.min()

    area_m_max2 = area_m_log.max(axis=1)
    area_m_norm2 = area_m_log / area_m_max2[:, None]

    # weight based on train_val data
    train_val_idx = (df_global['q_pr_global'] < 0.01) & (df_global['decoy'] == 0)
    cscores_pr = df_global['cscore_pr_global'].values
    W_cscore = cscores_pr[train_val_idx]

    # pytorch
    X_area1 = torch.tensor(area_m_norm1).to(param_g.gpu_id)
    X_area2 = torch.tensor(area_m_norm2).to(param_g.gpu_id)
    X_sa = torch.tensor(sa_m).to(param_g.gpu_id)
    W_cscore = torch.tensor(W_cscore).to(param_g.gpu_id)

    X_area1_train_val = X_area1[train_val_idx]
    X_area2_train_val = X_area2[train_val_idx]
    X_sa_train_val = X_sa[train_val_idx]
    dataset = TensorDataset(
        X_area1_train_val, X_area2_train_val, X_sa_train_val, W_cscore
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
    train_loader = DataLoader(train_set, batch_size=64, drop_last=True,
                              shuffle=True, generator=torch.Generator().manual_seed(42))
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
        for X_area1_batch, X_area2_batch, X_sa_batch, W_cscore in train_loader:
            optimizer.zero_grad()
            recon = model(X_area1_batch, X_area2_batch, X_sa_batch)
            loss = criterion(recon, X_area1_batch)
            loss = (loss * X_sa_batch).mean(dim=1)
            loss = (loss * W_cscore).mean()
            epoch_train_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_loss = epoch_train_loss / len(train_loader)

        # val
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for X_area1_val, X_area2_val, X_sa_val, W_cscore in val_loader:
                recon_val = model(X_area1_val, X_area2_val, X_sa_val)
                loss = criterion(recon_val, X_area1_val)
                loss = (loss * X_sa_val).mean(dim=1)
                loss = (loss * W_cscore).mean()
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
    pred_m = pred_m * area_m_log_max1
    # correct for log transformer
    good_idx = area_m_log != area_m_log.min()
    c = np.exp2(area_m_log[good_idx] - pred_m[good_idx]).flatten().mean()
    pred_m = np.exp2(pred_m)
    if is_log_correct:
        pred_m = pred_m * c

    # quant
    sa_sum = np.nansum(sa_m_v, axis=0)
    top_n_idx = np.argsort(sa_sum, axis=1)[:, -top_k_fg:]
    for run_idx, area_m in enumerate(area_m_v):
        # quant by raw or deep
        top_n_values = np.take_along_axis(area_m, top_n_idx, axis=1)
        pr_quant_raw = top_n_values.sum(axis=1)

        area_m_ae = pred_m[:, run_idx * n_ion: (run_idx + 1) * n_ion]
        top_n_values = np.take_along_axis(area_m_ae, top_n_idx, axis=1)
        pr_quant_deep = top_n_values.sum(axis=1)

        df_global['quant_pr_raw_' + str(run_idx)] = pr_quant_raw
        df_global['quant_pr_deep_' + str(run_idx)] = pr_quant_deep
        quant_pr_mix = np.sqrt(pr_quant_raw * pr_quant_deep)
        df_global['quant_pr_mix_' + str(run_idx)] = quant_pr_mix

    return df_global


def save_report_result(df_global, multi_ws=[], df_v=[]):
    logger.info('Saving report.parquet ...')
    if len(multi_ws) > 0:
        oname = 'report.parquet'
    if len(df_v) > 0:
        oname = 'report-second.tsv'

    df_out_v = []
    for ws_i in range(max(len(multi_ws), len(df_v))):
        if len(multi_ws):
            df = utils.read_from_pq(multi_ws[ws_i])
        else:
            df = df_v[ws_i]

        # merge global info: cscore_global, q_global, quant pr/pg
        cols = df_global.columns[~df_global.columns.str.startswith('quant_')]
        cols_quant_pr = [f'{x}_{ws_i}' for x in ['quant_pr_raw', 'quant_pr_deep', 'quant_pr_mix']]
        cols_quant_pg = [f'{x}_{ws_i}' for x in ['quant_pg_raw', 'quant_pg_deep', 'quant_pg_mix']]
        cols = cols.tolist() + cols_quant_pr + cols_quant_pg
        df = pd.merge(df, df_global[cols], on=['pr_id', 'decoy'], how='right')
        cols_new = ['quant_pr_raw', 'quant_pr_deep', 'quant_pr_mix']
        df = df.rename(columns=dict(zip(cols_quant_pr, cols_new)))
        cols_new = ['quant_pg_raw', 'quant_pg_deep', 'quant_pg_mix']
        df = df.rename(columns=dict(zip(cols_quant_pg, cols_new)))

        # cal q_pg_run based on assigned protein_groups
        df['cscore_pr_run'].fillna(0, inplace=True)
        df = fdr.cal_q_pg(df, param_g.q_cut_infer, 'run')

        # dtype
        df['pr_charge'] = df['pr_id'].str[-1].astype(np.int8)
        df['proteotypic'] = df['proteotypic'].astype(np.int8)
        cols_big = df.select_dtypes(include=[np.float64]).columns
        df[cols_big] = df[cols_big].astype(np.float32)

        # convert
        df = utils.convert_cols_to_diann(df, param_g.multi_ws[ws_i])
        df_out_v.append(df)
    df = pd.concat(df_out_v, ignore_index=True)
    df.to_parquet(param_g.dir_out_global / oname)