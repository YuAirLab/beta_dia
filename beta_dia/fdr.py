import os
os.environ["PYTHONWARNINGS"] = "ignore" # multiprocess

import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import VotingClassifier
from joblib import Parallel, delayed

import warnings
warnings.simplefilter("ignore")
warnings.filterwarnings(action='ignore', category=UserWarning)
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

from beta_dia import utils
from beta_dia import param_g
from beta_dia.log import Logger

try:
    # profile
    profile = lambda x: x
except NameError:
    profile = lambda x: x

logger = Logger.get_logger()

class SoftVotingMLPEnsemble:
    def __init__(self):
        param = (25, 20, 15, 10, 5)
        mlps = [MLPClassifier(max_iter=1,
                              warm_start=True,
                              early_stopping=False,
                              shuffle=True,
                              random_state=i,  # init weights and shuffle
                              learning_rate_init=0.003,
                              solver='adam',
                              batch_size=50,  # DIA-NN is 50
                              activation='relu',
                              hidden_layer_sizes=param) for i in range(12)]
        self.estimators_ = mlps
        self.estimators_init = [False for i in range(12)]

    def partial_fit(self, X, y):
        n_jobs = 1 if __debug__ else 12
        new_estimators = Parallel(n_jobs=n_jobs)(
            delayed(self._train_single)(self.estimators_[i], self.estimators_init[i], X, y)
            for i in range(len(self.estimators_))
        )
        self.estimators_ = new_estimators

    def _train_single(self, estimator, is_init, X, y):
        est = deepcopy(estimator)
        if not is_init:
            est.partial_fit(X, y, classes=[0, 1])
            self.first_fit = False
        else:
            est.partial_fit(X, y)
        return est

    def predict_proba(self, X):
        n_jobs = 1 if __debug__ else 12
        probas = Parallel(n_jobs=n_jobs)(
            delayed(estimator.predict_proba)(X)
            for estimator in self.estimators_
        )
        return np.mean(probas, axis=0)


def adjust_rubbish_q(df, batch_num):
    ids = df[(df['q_pr_run'] < 0.01) &
             (df['decoy'] == 0) &
             (df['group_rank'] == 1)].pr_id.nunique()
    ids = ids * batch_num
    if ids < 5000:
        param_g.rubbish_q_cut = 0.75
    else:
        param_g.rubbish_q_cut = param_g.rubbish_q_cut


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
    assert sum(col_idx) == 392
    # logger.info('cols num: {}'.format(sum(col_idx)))

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
    logger.info('scores items: {}'.format(sum(col_idx)))

    X = df.loc[:, col_idx].values
    assert X.dtype == np.float32
    y = 1 - df['decoy'].values  # targets is positives

    # select by train_nn_type: 1-hard, 2-easy, 3-cross
    scaler = preprocessing.StandardScaler()
    X = scaler.fit_transform(X) # no scale to Tree models

    # train
    n_pos, n_neg = sum(y == 1), sum(y == 0)
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
    model.fit(X, y)

    # pred
    cscore = model.predict_proba(X)[:, 1]
    df['cscore_pr_run'] = cscore

    # mirrors does not involve this
    df = cal_q_pr_core(df, 'run')
    return df


def cal_q_pr_second(df_input, batch_size, n_model, cols_start='score_'):
    col_idx = df_input.columns.str.startswith(cols_start)
    logger.info('cols num: {}'.format(sum(col_idx)))

    X = df_input.loc[:, col_idx].values
    assert X.dtype == np.float32
    y = 1 - df_input['decoy'].values  # targets is positives

    # select by train_nn_type: 1-hard, 2-easy, 3-cross
    scaler = preprocessing.StandardScaler()
    X = scaler.fit_transform(X)  # no scale to Tree models

    # training on group_rank == 1
    n_pos, n_neg = sum(y == 1), sum(y == 0)
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
        max_iter=5,
        shuffle=True,
        random_state=i,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=2,
    ) for i in range(n_model)]
    names = [f'mlp{i}' for i in range(n_model)]
    model = VotingClassifier(estimators=list(zip(names, mlps)),
                             voting='soft',
                             n_jobs=1 if __debug__ else 12)
    model.fit(X, y)
    cscore = model.predict_proba(X)[:, 1]

    df_input['cscore_pr_run'] = cscore

    # mirrors does not involve this
    df_input = cal_q_pr_core(df_input, 'run')
    return df_input


def cal_q_pr_NN_NN(df_input, batch_size, n_model, cols_start='score_'):
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
    preds = [clf.predict_proba(X_train)[:, 1] for clf in model.estimators_]
    preds = np.array(preds).T
    meta = MLPClassifier(max_iter=1,
                          shuffle=True,
                          random_state=42,  # init weights and shuffle
                          learning_rate_init=0.003,
                          solver='adam',
                          batch_size=batch_size,  # DIA-NN is 50
                          activation='relu',
                          hidden_layer_sizes=[32, 16, 4])
    meta.fit(preds, y_train)
    cscore = meta.predict_proba(preds)[:, 1]


    # mirrors does not involve this
    df_main = df_input[df_input['is_main']]
    df_other = df_input[~df_input['is_main']]

    df_main['cscore_pr_run'] = cscore
    df_main = cal_q_pr_core(df_main, 'run')
    df_other['q_pr_run'] = [1] * len(df_other) # valid cscore, invalid q value
    df = pd.concat([df_main, df_other], axis=0, ignore_index=True)

    return df


def cal_q_pr_kfold(df_input, batch_size, n_model, cols_start='score_'):
    df_input['cscore_pr_run'] = np.float32(0.)

    df_main = df_input[df_input['is_main']].reset_index(drop=True)
    df_other = df_input[~df_input['is_main']].reset_index(drop=True)
    df_main = df_main.sample(frac=1, random_state=42).reset_index(drop=True)

    col_idx = df_main.columns.str.startswith(cols_start)
    logger.info('cols num: {}'.format(sum(col_idx)))

    X = df_main.loc[:, col_idx].values
    assert X.dtype == np.float32
    y = 1 - df_main['decoy'].values  # targets is positives
    scaler = preprocessing.StandardScaler()
    X = scaler.fit_transform(X)

    # k-folder
    k = 5
    all_idx = np.arange(len(X))
    cscore_v = []
    for val_idx in np.array_split(all_idx, k):
        train_idx = np.setdiff1d(all_idx, val_idx)
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_val = X[val_idx]

        # training on group_rank == 1
        n_pos, n_neg = sum(y_train == 1), sum(y_train == 0)
        info = 'Training the model: {} pos, {} neg'.format(n_pos, n_neg)
        logger.info(info)

        # models
        param = (25, 20, 15, 10, 5)
        mlps = [MLPClassifier(max_iter=4,
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

        cscore = model.predict_proba(X_val)[:, 1]
        df_main.loc[val_idx, 'cscore_pr_run'] = cscore

    # mirrors does not involve this
    df_main = cal_q_pr_core(df_main, 'run')
    df_other['q_pr_run'] = [1] * len(df_other) # valid cscore, invalid q value
    df = pd.concat([df_main, df_other], axis=0, ignore_index=True)

    return df


def get_fake_decoy(df_decoy, n):
    df_fake = df_decoy.copy()
    df_fake = df_fake.sample(frac=1, random_state=42).reset_index(drop=True)

    col_idx = df_decoy.columns.str.startswith('score_')
    cols = df_decoy.columns[col_idx].tolist() + ['cscore_pr_run']

    x = df_decoy[cols].values
    y = df_fake[cols].values
    z = (x + y) / 2
    int_cols = np.all(np.isclose(x, np.round(x)), axis=0)
    int_cols = int_cols[np.newaxis, :]
    z = np.where(int_cols, np.round(z), z)

    df_fake[cols] = z
    df_fake['decoy'] = 2

    return df_fake.nlargest(n, 'cscore_pr_run')


def cal_fake_at_001(df_input):
    df = df_input[df_input['decoy'] != 1].reset_index(drop=True)

    df = df.sort_values(by='cscore_pr_run', ascending=False, ignore_index=True)
    target_num = (df.decoy == 0).cumsum()
    decoy_num = (df.decoy == 2).cumsum()

    target_num[target_num == 0] = 1
    df['q_fake'] = decoy_num / target_num
    df['q_fake'] = df['q_fake'][::-1].cummin()
    n = sum((df['q_fake'] < 0.01) & (df['decoy'] == 0))
    return n


def cal_q_pr_fake(df_input, batch_size, n_model, cols_start='score_'):
    df_main = df_input[df_input['is_main']].reset_index(drop=True)
    df_other = df_input[~df_input['is_main']].reset_index(drop=True)
    df_main = df_main.sample(frac=1, random_state=42).reset_index(drop=True)
    ids_001 = sum((df_main['q_pr_run'] < 0.01) & (df_main['decoy'] == 0))

    df_target = df_main[df_main['decoy'] == 0]
    df_decoy = df_main[df_main['decoy'] == 1]
    df_fake = get_fake_decoy(df_decoy, int(len(df_target) * 0.01))
    df_main = pd.concat([df_target, df_fake, df_decoy], axis=0, ignore_index=True)

    col_idx = df_main.columns.str.startswith(cols_start)
    logger.info('cols num: {}'.format(sum(col_idx)))

    X = df_main.loc[:, col_idx].values
    assert X.dtype == np.float32
    y = 1 - df_main['decoy'].values  # targets is positives
    y[y < 0] = 1
    scaler = preprocessing.StandardScaler()
    X = scaler.fit_transform(X)

    # model
    model = SoftVotingMLPEnsemble()
    model.partial_fit(X, y)
    cscore_best = model.predict_proba(X)[:, 1]
    df_main['cscore_pr_run'] = cscore_best
    ids_fake_best = sum(df_main.nlargest(ids_001, 'cscore_pr_run')['decoy'] == 2)
    logger.info(ids_fake_best)
    for i in range(5):
        model.partial_fit(X, y)
        cscore = model.predict_proba(X)[:, 1]
        df_main['cscore_pr_run'] = cscore
        ids_fake_now = sum(df_main.nlargest(ids_001, 'cscore_pr_run')['decoy'] == 2)
        logger.info(ids_fake_now)
        if ids_fake_now >= ids_fake_best:
            break
        else:
            cscore_best = cscore
            ids_fake_best = ids_fake_now

    df_main['cscore_pr_run'] = cscore_best
    df_main = df_main[df_main['decoy'] < 2].reset_index(drop=True)
    df_main = cal_q_pr_core(df_main, 'run')
    df_other['q_pr_run'] = [1] * len(df_other) # valid cscore, invalid q value
    df = pd.concat([df_main, df_other], axis=0, ignore_index=True)

    return df


def cal_q_pg(df_input_raw, q_pr_cut, run_or_global):
    '''
    for protein group q value calculation with IDPicker
    In reanalysis, the targets already have done the assign and q_pg_global
    But for decoys, they need to be reanalyzed.
    '''
    x = run_or_global
    df_na = df_input_raw[df_input_raw['cscore_pr_' + x].isna()]
    df_input = df_input_raw[~df_input_raw['cscore_pr_' + x].isna()]

    if 'strip_seq' not in df_input.columns:
        if 'simple_seq' not in df_input.columns:
            df_input['simple_seq'] = df_input['pr_id'].str[:-1].replace(
                ['C\(UniMod:4\)', 'M\(UniMod:35\)'], ['c', 'm'], regex=True
            )
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

    df_na['cscore_pg_' + x] = np.float32(0.)
    df_na['q_pg_' + x] = np.float32(1.)

    df_result = pd.concat([df_result, df_na], axis=0, ignore_index=True)

    return df_result
