import numpy as np
import pandas as pd
from functools import reduce
import pyarrow.parquet as pq

from beta_dia import param_g
from beta_dia import utils
from beta_dia.log import Logger
from beta_dia import fdr

try:
    profile
except NameError:
    profile = lambda x: x

logger = Logger.get_logger()


def drop_batches_mismatch(df):
    # remove decoy duplicates
    df_decoy = df[df['decoy'] == 1]
    idx_max = df_decoy.groupby('pr_id')['cscore_pr'].idxmax()
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
    idx_max = df_decoy.groupby('pr_id')['cscore_pr'].idxmax()
    df_decoy = df_decoy.loc[idx_max].reset_index(drop=True)

    # remove target duplicates
    df_target = df[df['decoy'] == 0]
    idx_max = df_target.groupby('pr_id')['cscore_pr'].idxmax()
    df_target = df_target.loc[idx_max].reset_index(drop=True)

    # remove decoy mismatch
    bad_idx = df_decoy['pr_id'].isin(df_target['pr_id'])
    df_decoy = df_decoy.loc[~bad_idx]

    df = pd.concat([df_target, df_decoy], axis=0, ignore_index=True)

    assert len(df) == df['pr_id'].nunique()

    return df


def get_global(multi_ws, lib, top_k_fg):
    '''
    A few items that this function will done:
    1. generate global df: [pr_id, decoy, cscore_pr_global, q_pr_global, strip_seq]
    2. assign prs to pro: [protein_group]
    3. cal q_pg_global: [q_pg_global]
    4. cross quant for pr: [quant_pr]
    Returns:
        df_global: global information, only targets, pr_id is unique
                   [pr_id, pg, cscore_pr/pg_global, q_pr/pg_global]
        df_cscore: for reanalysis, pr_id is not unique to cover each runs
                   [pr_id, decoy, cscore-0, ..., cscore-N]
        df_quant: pr_quant for each pr (targets in 5%) in each run
                   [pr_id, run_idx_0, run_idx_1, ..., run_idx-N]
    '''
    df_report_v, df_global_v, df_cscore_v, df_quant_v = [], [], [], []
    for ws_i, ws_single in enumerate(multi_ws):
        df_raw = utils.read_from_pq(ws_single)

        # for df_global
        df = df_raw[df_raw['is_main']]
        df = df[['pr_index', 'pr_id', 'simple_seq', 'decoy', 'cscore_pr']]
        df = df.rename(columns={'cscore_pr': 'score_cscore_' + str(ws_i)})
        df_global_v.append(df)

        # for df_cscore
        df = df_raw[['pr_id', 'cscore_pr', 'decoy']]
        df = df.rename(columns={'cscore_pr': 'score_cscore_' + str(ws_i)})
        df_cscore_v.append(df)

        # for df_quant
        cols_quant = ['fg_quant_' + str(i) for i in range(param_g.fg_num)]
        cols_sa = ['fg_sa_' + str(i) for i in range(param_g.fg_num)]
        cols = ['pr_id', 'decoy'] + cols_quant + cols_sa
        df = df_raw.loc[df_raw['decoy'] == 0, cols]
        df_quant_v.append(df)

    # df_cscore for run's reanalysis
    df_cscore = reduce(
        lambda x, y: pd.merge(x, y, on=['pr_id', 'decoy'], how='outer'),
        df_cscore_v
    )
    ids_num = df_cscore[['pr_id', 'decoy']].drop_duplicates().shape[0]
    assert ids_num == len(df_cscore) # save each item to cover each run
    df_cscore = df_cscore.fillna(0)

    # cal q_pr_global by maximum cscore method
    df_global = reduce(
        lambda x, y: pd.merge(
            x, y,
            on=['pr_id', 'simple_seq', 'decoy', 'pr_index'],
            how='outer'
        ),
        df_global_v
    )
    df_global = df_global.fillna(0.)
    cscore_cols = df_global.columns.str.startswith('score_cscore_')
    df_global['cscore_pr'] = df_global.loc[:, cscore_cols].max(axis=1)
    df_global = drop_runs_mismatch(df_global)
    df_global = fdr.cal_q_pr_core(df_global, 'cscore_pr')

    logger.info(f'Merge {len(multi_ws)} .parquet files resulting in: ')
    utils.print_ids(df_global, 0.05)

    # assign and q value for protein group
    df_global = lib.assign_proteins(df_global)
    df_global['strip_seq'] = df_global['simple_seq'].str.upper()
    df_global = fdr.cal_q_pg_prod(df_global, 0.05)
    utils.print_ids(df_global, 0.05, level='pg')

    # rename df_global
    df_global = df_global.rename(columns={
        'q_pr': 'q_pr_global',
        'q_pg': 'q_pg_global',
        'cscore_pr': 'cscore_pr_global',
        'cscore_pg': 'cscore_pg_global',
    })
    df_global = df_global[df_global['decoy'] == 0]
    cols = ['pr_id', 'proteotypic',
            'protein_id', 'protein_name', 'protein_group',
            'cscore_pr_global', 'q_pr_global',
            'cscore_pg_global', 'q_pg_global']
    df_global = df_global[cols].reset_index(drop=True)

    # cross quant
    df_quant = quant_pr_cross(df_global, df_quant_v, top_k_fg)

    return df_global, df_cscore, df_quant


def quant_pr_cross(df_global, df_quant_v, top_k_fg):
    # df_global, only targets, pr_id is unique
    prs_target = df_global[(df_global['q_pr_global'] < 0.05)]['pr_id']
    assert prs_target.nunique() == len(prs_target)

    sa_m_v, area_m_v = [], []
    for ws_i, df in enumerate(df_quant_v):
        df = df[df['pr_id'].isin(prs_target) & (df['decoy'] == 0)]

        df = df.set_index('pr_id').reindex(prs_target).reset_index()

        cols_sa = ['fg_sa_' + str(i) for i in range(param_g.fg_num)]
        sa_m = df[cols_sa].values
        sa_m_v.append(sa_m)

        cols_quant = ['fg_quant_' + str(i) for i in range(param_g.fg_num)]
        area_m = df[cols_quant].values
        area_m_v.append(area_m)

    # find best fg ions cross runs
    sa_sum = np.nansum(sa_m_v, axis=0)
    top_n_idx = np.argsort(sa_sum, axis=1)[:, -top_k_fg:]

    df = pd.DataFrame({'pr_id': prs_target})
    for run_idx, area_m in enumerate(area_m_v):
        top_n_values = np.take_along_axis(area_m, top_n_idx, axis=1)
        pr_quant = top_n_values.sum(axis=1)
        # sometimes global selection leads to zero for specific run
        pr_quant[(pr_quant <= 0) | np.isnan(pr_quant)] = pr_quant[pr_quant > 0].min()
        df['run_idx_' + str(run_idx)] = pr_quant

    return df
