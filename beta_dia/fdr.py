import os
os.environ["PYTHONWARNINGS"] = "ignore" # multiprocess

import numpy as np
import pandas as pd
import networkx as nx
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import VotingClassifier

import warnings
warnings.simplefilter("ignore")
warnings.filterwarnings(action='ignore', category=UserWarning)
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

from beta_dia import utils
from beta_dia import param_g
from beta_dia.log import Logger

try:
    profile
except NameError:
    profile = lambda x: x

logger = Logger.get_logger()

def cal_q_pr_core(df, score_col):
    df = df.sort_values(by=score_col, ascending=False, ignore_index=True)
    target_num = (df.decoy == 0).cumsum()
    decoy_num = (df.decoy == 1).cumsum()

    target_num[target_num == 0] = 1
    df['q_pr'] = decoy_num / target_num

    df['q_pr'] = df['q_pr'][::-1].cummin()
    return df


def cal_q_pro_prod(df_input, q_pr_cut):
    '''
    for protein q value calculation using proteotypic peptides
    '''
    df = df_input.copy()

    df = df[df['proteotypic'] == 1].reset_index(drop=True)
    idx_max = df.groupby('strip_seq')['cscore_pr'].idxmax()
    df = df.loc[idx_max].reset_index(drop=True)

    df_target = df[(df['decoy'] == 0) &
                   (df['q_pr'] < q_pr_cut)].reset_index(drop=True)
    df_decoy = df[(df['decoy'] == 1) &
                  (df['q_pr'] < q_pr_cut)].reset_index(drop=True)

    # target
    df_target = df_target.groupby('protein_id')['cscore_pr'].apply(
        lambda g: 1 - (1 - g).prod()
    )
    df_target = df_target.reset_index()
    df_target['decoy'] = 0

    # decoy
    df_decoy = df_decoy.groupby('protein_id')['cscore_pr'].apply(
        lambda g: 1 - (1 - g).prod()
    )
    df_decoy = df_decoy.reset_index()
    df_decoy['decoy'] = 1

    # q
    df = pd.concat([df_target, df_decoy])
    df = df.rename(columns={'cscore_pr': 'cscore_pro'})
    df = df.sort_values(by='cscore_pro', ascending=False, ignore_index=True)
    target_num = (df.decoy == 0).cumsum()
    decoy_num = (df.decoy == 1).cumsum()

    target_num[target_num == 0] = 1
    df['q_pro'] = decoy_num / target_num

    df['q_pro'] = df['q_pro'][::-1].cummin()
    del df['decoy']

    df = df_input.merge(df, on='protein_id', how='left').reset_index(drop=True)
    df['cscore_pro'] = df['cscore_pro'].fillna(0.)
    df['q_pro'] = df['q_pro'].fillna(1)

    return df


def cal_q_pg_prod(df_input, q_pr_cut):
    '''
    for protein group q value calculation with IDPicker
    '''
    # seq to strip_seq
    df_pep_score = df_input[['strip_seq', 'cscore_pr']].copy()
    idx_max = df_pep_score.groupby(['strip_seq'])['cscore_pr'].idxmax()
    df_pep_score = df_pep_score.loc[idx_max].reset_index(drop=True)

    # target: first assign, then score
    df_target = df_input[(df_input['decoy'] == 0) &
                         (df_input['q_pr'] < q_pr_cut)].copy()
    df_target = assign_pep_to_pg(df_target)
    df_target = df_target.merge(df_pep_score, on='strip_seq')
    df_target = df_target.groupby('protein_group').agg(
        {
            'cscore_pr': lambda g: 1 - (1 - g).prod(),
            # 'cscore_pr': lambda g: g.nlargest(1).sum(),
            'strip_seq': lambda g: list(g)}
    ).reset_index()
    df_target = df_target.rename(columns={'cscore_pr': 'cscore_pg'})
    df_target['decoy'] = 0

    # decoy
    df_decoy = df_input[(df_input['decoy'] == 1) &
                         (df_input['q_pr'] < q_pr_cut)].copy()
    df_decoy = assign_pep_to_pg(df_decoy)
    df_decoy = df_decoy.merge(df_pep_score, on='strip_seq')
    df_decoy = df_decoy.groupby('protein_group').agg(
        {
            'cscore_pr': lambda g: 1 - (1 - g).prod(),
            # 'cscore_pr': lambda g: g.nlargest(1).sum(),
            'strip_seq': lambda g: list(g)}
    ).reset_index()
    df_decoy = df_decoy.rename(columns={'cscore_pr': 'cscore_pg'})
    df_decoy['decoy'] = 1

    # q
    df = pd.concat([df_target, df_decoy])
    df = df.sort_values(by='cscore_pg', ascending=False, ignore_index=True)
    target_num = (df.decoy == 0).cumsum()
    decoy_num = (df.decoy == 1).cumsum()

    target_num[target_num == 0] = 1
    df['q_pg'] = decoy_num / target_num

    df['q_pg'] = df['q_pg'][::-1].cummin()

    # explode, get df: [strip_seq, protein_group, q_pg]
    del df['decoy']
    pep_num = df['strip_seq'].apply(len).values
    peptide_v = df['strip_seq'].explode().tolist()
    df = df.loc[np.repeat(df.index, pep_num)]
    df = df.reset_index(drop=True)
    df['strip_seq'] = peptide_v

    # result
    df = df_input.merge(df, on='strip_seq', how='left').reset_index(drop=True)
    not_in_range = df['q_pg'].isna()
    df.loc[not_in_range, 'protein_group'] = df.loc[not_in_range, 'protein_id']
    df.loc[not_in_range, 'cscore_pg'] = 0.
    df.loc[not_in_range, 'q_pg'] = 1

    return df


def cal_q_pg_after_cross(df_input, lib, q_pr_cut):
    '''
    for protein group q value calculation with IDPicker
    In reanalysis, the targets already have done the assign and q_pg_global
    But for decoys, they need to be reanalyzed.
    '''
    # seq to strip_seq
    df_pep_score = df_input[['strip_seq', 'cscore_pr']].copy()
    idx_max = df_pep_score.groupby(['strip_seq'])['cscore_pr'].idxmax()
    df_pep_score = df_pep_score.loc[idx_max].reset_index(drop=True)

    # target: first assign, then score
    df_target = df_input[(df_input['decoy'] == 0) &
                         (df_input['q_pr'] < q_pr_cut)].copy()
    df_target = df_target[['strip_seq', 'protein_group']]
    df_target = df_target.merge(df_pep_score, on='strip_seq')
    df_target = df_target.groupby('protein_group').agg(
        {
            'cscore_pr': lambda g: 1 - (1 - g).prod(),
            'strip_seq': lambda g: list(g)}
    ).reset_index()
    df_target = df_target.rename(columns={'cscore_pr': 'cscore_pg'})
    df_target['decoy'] = 0

    # decoy
    df_decoy = df_input[(df_input['decoy'] == 1) &
                         (df_input['q_pr'] < q_pr_cut)].copy()
    df_decoy = lib.assign_proteins(df_decoy)
    df_decoy = assign_pep_to_pg(df_decoy)

    df_decoy = df_decoy[['strip_seq', 'protein_group']]
    df_decoy = df_decoy.merge(df_pep_score, on='strip_seq')
    df_decoy = df_decoy.groupby('protein_group').agg(
        {
            'cscore_pr': lambda g: 1 - (1 - g).prod(),
            'strip_seq': lambda g: list(g)}
    ).reset_index()
    df_decoy = df_decoy.rename(columns={'cscore_pr': 'cscore_pg'})
    df_decoy['decoy'] = 1

    # q
    df = pd.concat([df_target, df_decoy])
    df = df.sort_values(by='cscore_pg', ascending=False, ignore_index=True)
    target_num = (df.decoy == 0).cumsum()
    decoy_num = (df.decoy == 1).cumsum()

    target_num[target_num == 0] = 1
    df['q_pg'] = decoy_num / target_num

    df['q_pg'] = df['q_pg'][::-1].cummin()

    # explode, get df: [strip_seq, protein_group, q_pg]
    pep_num = df['strip_seq'].apply(len).values
    peptide_v = df['strip_seq'].explode().tolist()
    df = df.loc[np.repeat(df.index, pep_num)]
    df = df.reset_index(drop=True)
    df['strip_seq'] = peptide_v

    # merge q_pg to target, merge protein group/q_pg to decoy
    df_target = df_input[df_input['decoy'] == 0]
    df_q = df.loc[df['decoy'] == 0, ['strip_seq', 'decoy', 'cscore_pg', 'q_pg']]
    df_target = df_target.merge(df_q, on=['strip_seq', 'decoy'], how='left')

    df_decoy = df_input[df_input['decoy'] == 1]
    del df_decoy['protein_group']
    df_q = df.loc[df['decoy'] == 1, ['strip_seq', 'decoy', 'protein_group', 'cscore_pg', 'q_pg']]
    df_decoy = df_decoy.merge(df_q, on=['strip_seq', 'decoy'], how='left').reset_index(drop=True)

    df = pd.concat([df_target, df_decoy], ignore_index=True)

    not_in_range = df['q_pg'].isna()
    df.loc[not_in_range, 'protein_group'] = df.loc[not_in_range, 'protein_id']
    df.loc[not_in_range, 'cscore_pg'] = 0.
    df.loc[not_in_range, 'q_pg'] = 1

    return df


def greedy_bipartite_vertex_cover(graph):
    graph = nx.freeze(graph)
    graph = nx.Graph(graph)
    left_nodes = [node for node, data in graph.nodes(data=True) if
                  data["bipartite"] == 0]
    right_nodes = [node for node, data in graph.nodes(data=True) if
                   data["bipartite"] == 1]
    protein_v, peptide_v = [], []

    while right_nodes:
        # select nodes with most edges
        node = max(left_nodes, key=graph.degree)
        neighbors = list(graph.neighbors(node))
        protein_v.append(node)
        peptide_v.append(neighbors)

        graph.remove_nodes_from([node] + neighbors)
        left_nodes = [node for node, data in graph.nodes(data=True) if
                      data["bipartite"] == 0]
        right_nodes = [node for node, data in graph.nodes(data=True) if
                       data["bipartite"] == 1]
    return protein_v, peptide_v


def assign_pep_to_pg(df):
    df = df[['protein_id', 'strip_seq']].copy()
    df['protein_id'] = df['protein_id'].str.split(';')
    proteins = df['protein_id'].explode().values
    protein_num = df['protein_id'].apply(len)

    df = df.loc[np.repeat(df.index, protein_num)]
    df = df.reset_index(drop=True)
    df['protein_id'] = proteins

    # protein meta
    df_protein = df.groupby('protein_id', sort=False)['strip_seq'].agg(
        set)
    df_protein = df_protein.reset_index()
    df_protein['strip_seq'] = df_protein['strip_seq'].apply(tuple)
    df_protein = df_protein.groupby('strip_seq', sort=False)[
        'protein_id'].agg(set)
    df_protein = df_protein.reset_index()

    # corresponding
    df_protein['Protein.Meta'] = df_protein['protein_id'].str.join(';')
    proteins = df_protein['protein_id'].explode().values
    protein_num = df_protein['protein_id'].apply(len)
    df_protein = df_protein.loc[
        np.repeat(df_protein.index, protein_num)].reset_index(drop=True)
    df_protein['Protein'] = proteins
    df_protein = df_protein[['Protein', 'Protein.Meta']]

    df_protein.set_index('Protein', inplace=True)

    # from 1 vs. 1 to meta vs. meta
    df['Protein.Meta'] = df_protein.loc[df['protein_id']][
        'Protein.Meta'].values
    df['Peptide.Meta'] = df['strip_seq']
    df = df[['Protein.Meta', 'Peptide.Meta']]
    df = df.drop_duplicates().reset_index(drop=True)

    # graph
    graph = nx.Graph()
    graph.add_nodes_from(df['Protein.Meta'], bipartite=0)
    graph.add_nodes_from(df['Peptide.Meta'], bipartite=1)
    graph.add_edges_from(df.values)

    # assign
    protein_v, peptide_v = [], []
    subgraphs = list(nx.connected_components(graph))
    for subgraph in subgraphs:
        subgraph = graph.subgraph(subgraph)
        proteins, peptides = greedy_bipartite_vertex_cover(subgraph)
        protein_v.extend(proteins)
        peptide_v.extend(peptides)

    df = pd.DataFrame({'strip_seq': peptide_v,
                       'protein_group': protein_v})
    pep_num = df['strip_seq'].apply(len).values
    peptide_v = df['strip_seq'].explode().tolist()
    df = df.loc[np.repeat(df.index, pep_num)]
    df = df.reset_index(drop=True)
    df['strip_seq'] = peptide_v

    return df


def adjust_rubbish_q(df, batch_num):
    ids = df[(df['q_pr'] < 0.01) &
             (df['decoy'] == 0) &
             (df['group_rank'] == 1)].pr_id.nunique()
    ids = ids * batch_num
    if ids < 5000:
        rubbish_cut = 0.75
    else:
        rubbish_cut = param_g.rubbish_q_cut
    return rubbish_cut


def filter_by_q_cut(df, q_cut):
    if q_cut < 1:
        # main
        df_main = df[(df['q_pr'] < q_cut)]
        df_main['is_main'] = True
        targets = df_main[df_main['decoy'] == 0]['pr_id']

        # other1: main's corresponding decoy not in main
        df_other1 = df[(df['pr_root'].isin(targets)) &
                      (df['decoy'] == 1) &
                      (~df['pr_id'].isin(df_main['pr_id']))
                      ]
        df_other1['is_main'] = False

        # other2: high fdr prs and their corresponding decoys
        # caution: they may be in the df_main.decoy
        other2_targets = df[(df['q_pr'] > q_cut) &
                           (df['q_pr'] < q_cut + 0.2) &
                           (df['decoy'] == 0)]['pr_id']
        df_other2 = df[df['pr_root'].isin(other2_targets) &
                       ~df['pr_id'].isin(df_main['pr_id']) &
                       ~df['pr_id'].isin(df_other1['pr_id'])
                       ]
        df_other2['is_main'] = False

        df = pd.concat([df_main, df_other1, df_other2], ignore_index=True)

    return df


@profile
def cal_q_pr_batch(df, batch_size, n_model, model_trained=None, scaler=None):
    col_idx = df.columns.str.startswith('score_')
    logger.info('cols num: {}'.format(sum(col_idx)))

    X = df.loc[:, col_idx].values
    assert X.dtype == np.float32
    y = 1 - df['decoy'].values  # targets is positives
    if scaler is None:
        scaler = preprocessing.StandardScaler()
        X = scaler.fit_transform(X) # no scale to Tree models
    else:
        X = scaler.transform(X)

    # train
    if model_trained is None: # the first batch
        decoy_deeps = df.loc[df['decoy'] == 1, 'score_big_deep_pre'].values
        decoy_m, decoy_u = np.mean(decoy_deeps), np.std(decoy_deeps)
        good_cut = min(0.5, decoy_m + 1.5 * decoy_u)
        logger.info(f'Training with big_score_cut: {good_cut:.2f}')
        train_idx = (df['group_rank'] == 1) & (df['score_big_deep_pre'] > good_cut)
        X_train = X[train_idx]
        y_train = y[train_idx]

        n_pos, n_neg = sum(y_train == 1), sum(y_train == 0)
        info = 'Training the NN model: {} pos, {} neg'.format(n_pos, n_neg)
        logger.info(info)

        param = (25, 20, 15, 10, 5)
        mlps = [MLPClassifier(max_iter=1,
                              shuffle=True,
                              random_state=i,  # init weights and shuffle
                              learning_rate_init=0.003,
                              solver='adam',
                              batch_size=batch_size,  # DIA-NN is 50
                              activation='relu',
                              hidden_layer_sizes=param) for i in range(n_model)]
        names = [f'mlp{i}' for i in range(n_model)]
        model = VotingClassifier(estimators=list(zip(names, mlps)),
                                 voting='soft',
                                 n_jobs=1 if __debug__ else n_model)
        model.fit(X_train, y_train)

        n_pos, n_neg = sum(y == 1), sum(y == 0)
        info = 'Predicting by the NN model: {} pos, {} neg'.format(n_pos, n_neg)
        logger.info(info)
        cscore = model.predict_proba(X)[:, 1]
    else:
        model = model_trained
        n_pos, n_neg = sum(y == 1), sum(y == 0)
        info = 'Predicting by the NN model: {} pos, {} neg'.format(n_pos, n_neg)
        logger.info(info)
        cscore = model.predict_proba(X)[:, 1]

    df['cscore_pr'] = cscore

    # group rank
    group_size = df.groupby('pr_id', sort=False).size()
    group_size_cumsum = np.concatenate([[0], np.cumsum(group_size)])
    group_rank = utils.cal_group_rank(df.cscore_pr.values, group_size_cumsum)
    df['group_rank'] = group_rank
    df = df.loc[group_rank == 1]

    df = cal_q_pr_core(df, score_col='cscore_pr')

    return df, model, scaler


@profile
def cal_q_pr_first(df, batch_size, n_model):
    col_idx = df.columns.str.startswith('score_')
    logger.info('cols num: {}'.format(sum(col_idx)))

    X = df.loc[:, col_idx].values
    assert X.dtype == np.float32
    y = 1 - df['decoy'].values  # targets is positives

    # select by train_nn_type: 1-hard, 2-easy, 3-cross
    idx = df['is_main'].values
    X_train = X[idx]
    y_train = y[idx]
    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train) # no scale to Tree models
    X = scaler.transform(X)

    # train
    n_pos, n_neg = sum(y_train == 1), sum(y_train == 0)
    info = 'Training the NN model: {} pos, {} neg'.format(n_pos, n_neg)
    logger.info(info)

    param = (25, 20, 15, 10, 5)
    mlps = [MLPClassifier(max_iter=1,
                          shuffle=True,
                          random_state=i,  # init weights and shuffle
                          learning_rate_init=0.003,
                          solver='adam',
                          batch_size=batch_size,  # DIA-NN is 50
                          activation='relu',
                          hidden_layer_sizes=param) for i in range(n_model)]
    names = [f'mlp{i}' for i in range(n_model)]
    model = VotingClassifier(estimators=list(zip(names, mlps)),
                             voting='soft',
                             n_jobs=1 if __debug__ else n_model)
    model.fit(X_train, y_train)

    # pred
    cscore = model.predict_proba(X)[:, 1]
    df['cscore_pr'] = cscore

    # mirrors does not involve this
    df_main = df[df['is_main']]
    df_other = df[~df['is_main']]

    df_main = cal_q_pr_core(df_main, score_col='cscore_pr')
    df_other['q_pr'] = 1
    df = pd.concat([df_main, df_other], axis=0, ignore_index=True)

    return df


def cal_q_pr_second(df_input, batch_size, n_model):
    col_idx = df_input.columns.str.startswith('score_')
    logger.info('cols num: {}'.format(sum(col_idx)))

    X = df_input.loc[:, col_idx].values
    assert X.dtype == np.float32
    y = 1 - df_input['decoy'].values  # targets is positives

    # select by train_nn_type: 1-hard, 2-easy, 3-cross
    idx = df_input['is_main'].values
    X_train = X[idx]
    y_train = y[idx]
    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)  # no scale to Tree models
    X = scaler.transform(X)

    # training on group_rank == 1
    n_pos, n_neg = sum(y_train == 1), sum(y_train == 0)
    info = 'Training the model: {} pos, {} neg'.format(n_pos, n_neg)
    logger.info(info)

    # models
    param = (25, 20, 15, 10, 5)
    mlps = [MLPClassifier(
        hidden_layer_sizes=param,
        activation='relu',
        solver='adam',
        alpha=0.0001, # L2 regular loss, default=0.0001
        batch_size=batch_size,
        learning_rate_init=0.001, # default
        max_iter=10,
        shuffle=True,
        random_state=i,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=5,
    ) for i in range(n_model)]
    names = [f'mlp{i}' for i in range(n_model)]
    model = VotingClassifier(estimators=list(zip(names, mlps)),
                             voting='soft',
                             n_jobs=1 if __debug__ else 12)
    model.fit(X_train, y_train)
    cscore = model.predict_proba(X)[:, 1]

    df_input['cscore_pr'] = cscore

    # mirrors does not involve this
    df_main = df_input[df_input['is_main']]
    df_other = df_input[~df_input['is_main']]
    df_main = cal_q_pr_core(df_main, score_col='cscore_pr')
    df_other['q_pr'] = [1] * len(df_other) # valid cscore, invalid q value
    df = pd.concat([df_main, df_other], axis=0, ignore_index=True)

    return df


def cal_q_pro_pg(df_top, q_pr_cut):
    df_top['strip_seq'] = df_top['simple_seq'].str.upper()

    df_fdr = cal_q_pro_prod(df_top, q_pr_cut)
    df_fdr = cal_q_pg_prod(df_fdr, q_pr_cut)

    df_result = df_fdr[(df_fdr['decoy'] == 0) &
                       (df_fdr['group_rank'] == 1)].reset_index(drop=True)

    for q in [0.001, 0.01, 0.02, 0.03, 0.05]:

        num_pr = df_result[df_result['q_pr'] <= q]['pr_id'].nunique()
        num_pro = df_result[df_result['q_pro'] <= q]['protein_id'].nunique()
        num_pg = df_result[df_result['q_pg'] <= q]['protein_group'].nunique()

        logger.info(f'Fdr-{q}: #pr-{num_pr}, #pro-{num_pro}, #pg-{num_pg}')

        utils.cal_acc_recall(param_g.ws_single, df_result,
                             diann_q_pr=q, diann_q_pro=q, diann_q_pg=q,
                             alpha_q_pr=q, alpha_q_pro=q, alpha_q_pg=q)

    return df_result
