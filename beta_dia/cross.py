import numpy as np
import pandas as pd
from functools import reduce

from beta_dia import param_g
from beta_dia import utils
from beta_dia.log import Logger
from beta_dia import fdr
from beta_dia import assemble

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
    cols = ['pr_index', 'pr_id', 'simple_seq', 'decoy', 'cscore_pr_run', 'is_main']
    cols_quant = ['fg_quant_' + str(i) for i in range(param_g.fg_num)]
    cols_sa = ['fg_sa_' + str(i) for i in range(param_g.fg_num)]
    cols = cols + cols_quant + cols_sa

    df_v = []
    for ws_i, ws_single in enumerate(multi_ws):
        df_raw = utils.read_from_pq(ws_single, cols)
        # global only considers is_main
        # non-main used for the recall when in reanalysis
        df = df_raw[df_raw['is_main']]
        del df['is_main']
        df = df.rename(columns={'cscore_pr_run': 'cscore_pr_run_' + str(ws_i)})
        cols_quant_long = ['run_' + str(ws_i) + '_' + x for x in cols_quant]
        df = df.rename(columns=dict(zip(cols_quant, cols_quant_long)))
        cols_sa_long = ['run_' + str(ws_i) + '_' + x for x in cols_sa]
        df = df.rename(columns=dict(zip(cols_sa, cols_sa_long)))
        df_v.append(df)

    # merge, a row is a pr
    df_global = reduce(
        lambda x, y: pd.merge(
            x, y,
            on=['pr_id', 'simple_seq', 'decoy', 'pr_index'],
            how='outer'
        ),
        df_v
    )
    df_global = df_global.fillna(0.)

    # cscore_pr_global
    cscore_cols = df_global.columns.str.startswith('cscore_pr_run_')
    df_global['cscore_pr_global'] = df_global.loc[:, cscore_cols].max(axis=1)

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

    # assemble: proteotypic, protein_id, protein_name, protein_group
    # cscore_pg_global, q_pg_global
    df_global = lib.assign_proteins(df_global)
    df_global = assemble.assemble_to_pg(df_global, param_g.q_cut_infer, 'global')
    df_global = fdr.cal_q_pg(df_global, param_g.q_cut_infer, 'global')

    # cross quant
    df_global = quant_pr_cross(df_global, top_k_fg)

    # log
    logger.info(f'Merge {len(multi_ws)} .parquet files resulting in first global: ')
    utils.print_ids(df_global, 0.05, pr_or_pg='pr', run_or_global='global')
    utils.print_ids(df_global, 0.05, pr_or_pg='pg', run_or_global='global')

    # return
    df_global = df_global.drop(['pr_index', 'simple_seq'], axis=1)
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


def quant_pr_cross(df_global, top_k_fg):
    import re
    n = max(int(m.group(1)) for col in df_global.columns if (m := re.search(r'run_(\d+)', col)))
    sa_m_v, area_m_v = [], []
    for wi in range(n + 1):
        cols_sa = ['run_' + str(wi) + '_' + 'fg_sa_' + str(i) for i in range(param_g.fg_num)]
        sa_m = df_global[cols_sa].values
        sa_m_v.append(sa_m)

        cols_quant = ['run_' + str(wi) + '_' + 'fg_quant_' + str(i) for i in range(param_g.fg_num)]
        area_m = df_global[cols_quant].values
        area_m_v.append(area_m)

    # find the best fg ions cross runs
    sa_sum = np.nansum(sa_m_v, axis=0)
    top_n_idx = np.argsort(sa_sum, axis=1)[:, -top_k_fg:]

    for run_idx, area_m in enumerate(area_m_v):
        top_n_values = np.take_along_axis(area_m, top_n_idx, axis=1)
        pr_quant = top_n_values.sum(axis=1)
        # sometimes global selection leads to zero for specific run
        pr_quant[(pr_quant <= 0) | np.isnan(pr_quant)] = pr_quant[pr_quant > 0].min()
        df_global['quant_pr_' + str(run_idx)] = pr_quant

    return df_global
