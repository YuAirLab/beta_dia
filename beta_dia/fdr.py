import os
os.environ["PYTHONWARNINGS"] = "ignore" # multiprocess

import numpy as np
import pandas as pd
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

def adjust_rubbish_q(df, batch_num):
    ids = df[(df['q_pr_run'] < 0.01) &
             (df['decoy'] == 0) &
             (df['group_rank'] == 1)].pr_id.nunique()
    ids = ids * batch_num
    if ids < 5000:
        rubbish_cut = 0.75
    else:
        rubbish_cut = param_g.rubbish_q_cut
    return rubbish_cut


def filter_by_q_cut(df, q_cut):
    # df has be sorted by the descending order of cscore_pr
    # add additional prs to save
    if q_cut < 1:
        df_target = df[df['decoy'] == 0]
        df_decoy = df[df['decoy'] == 1]

        target_num = (df_target['q_pr_run'] < q_cut).sum() * param_g.n_attached
        decoy_num = (df_decoy['q_pr_run'] < q_cut).sum() * param_g.n_attached

        df_target = df_target.iloc[:target_num, :]
        df_decoy = df_decoy.iloc[:decoy_num, :]

        df = pd.concat([df_target, df_decoy], ignore_index=True)
        df['is_main'] = False
        df.loc[df['q_pr_run'] < q_cut, 'is_main'] = True

    return df


def cal_q_pr_core(df, run_or_global):
    col_score = 'cscore_pr_' + run_or_global
    col_out = 'q_pr_' + run_or_global

    df = df.sort_values(by=col_score, ascending=False, ignore_index=True)
    target_num = (df.decoy == 0).cumsum()
    decoy_num = (df.decoy == 1).cumsum()

    target_num[target_num == 0] = 1
    df[col_out] = decoy_num / target_num

    df[col_out] = df[col_out][::-1].cummin()
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

    df['cscore_pr_run'] = cscore

    # group rank
    group_size = df.groupby('pr_id', sort=False).size()
    group_size_cumsum = np.concatenate([[0], np.cumsum(group_size)])
    group_rank = utils.cal_group_rank(df['cscore_pr_run'].values, group_size_cumsum)
    df['group_rank'] = group_rank
    df = df.loc[group_rank == 1]

    df = cal_q_pr_core(df, 'run')

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
    df['cscore_pr_run'] = cscore

    # mirrors does not involve this
    df_main = df[df['is_main']]
    df_other = df[~df['is_main']]

    df_main = cal_q_pr_core(df_main, 'run')
    df_other['q_pr_run'] = 1
    df = pd.concat([df_main, df_other], axis=0, ignore_index=True)

    return df


def cal_q_pr_second(df_input, batch_size, n_model, cols_start='score_'):
    col_idx = df_input.columns.str.startswith(cols_start)
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

    df_input['cscore_pr_run'] = cscore

    # mirrors does not involve this
    df_main = df_input[df_input['is_main']]
    df_other = df_input[~df_input['is_main']]
    df_main = cal_q_pr_core(df_main, 'run')
    df_other['q_pr_run'] = [1] * len(df_other) # valid cscore, invalid q value
    df = pd.concat([df_main, df_other], axis=0, ignore_index=True)

    return df


def cal_q_pg(df_input, q_pr_cut, run_or_global):
    '''
    for protein group q value calculation with IDPicker
    In reanalysis, the targets already have done the assign and q_pg_global
    But for decoys, they need to be reanalyzed.
    '''
    x = run_or_global
    if 'strip_seq' not in df_input.columns:
        df_input['strip_seq'] = df_input['simple_seq'].str.upper()

    # seq to strip_seq
    df_pep_score = df_input[['strip_seq', 'cscore_pr_' + x]].copy()
    idx_max = df_pep_score.groupby(['strip_seq'])['cscore_pr_' + x].idxmax()
    df_pep_score = df_pep_score.loc[idx_max].reset_index(drop=True)

    # row by protein group
    df = df_input[df_input['q_pr_' + x] < q_pr_cut]
    df = df[['strip_seq', 'protein_group', 'decoy']]
    df = df.drop_duplicates().reset_index(drop=True)
    df = df.merge(df_pep_score, on='strip_seq')
    df = df.groupby(by=['protein_group', 'decoy']).agg(
        {
            ('cscore_pr_' + x): lambda g: 1 - (1 - g).prod(),
            # ('cscore_pr_' + x): lambda g: g.nlargest(1).sum(),
            'strip_seq': lambda g: list(g)}
    ).reset_index()
    df = df.rename(columns={('cscore_pr_' + x): ('cscore_pg_' + x)})

    # q
    df = df.sort_values(by=('cscore_pg_' + x), ascending=False, ignore_index=True)
    target_num = (df.decoy == 0).cumsum()
    decoy_num = (df.decoy == 1).cumsum()
    target_num[target_num == 0] = 1
    df['q_pg_' + x] = decoy_num / target_num
    df['q_pg_' + x] = df['q_pg_' + x][::-1].cummin()

    df = df[['protein_group', 'decoy', 'cscore_pg_' + x, 'q_pg_' + x]]

    # return
    df_result = df_input.merge(df, on=['protein_group', 'decoy'], how='left')
    not_in_range = df_result['q_pg_' + x].isna()
    df_result.loc[not_in_range, 'cscore_pg_' + x] = 0.
    df_result.loc[not_in_range, 'q_pg_' + x] = 1

    return df_result
