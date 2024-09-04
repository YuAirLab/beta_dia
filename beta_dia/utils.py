import argparse
import gc
import warnings
from pathlib import Path

from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)

import cupy as cp
import numpy as np
import pandas as pd
import torch
from numba import cuda, jit, prange

from beta_dia import param_g
from beta_dia.log import Logger

# from xgboost.sklearn import XGBClassifier


try:
    profile
except NameError:
    profile = lambda x: x

logger = Logger.get_logger()

def release_gpu_scans(map_gpu):
    try:
        del map_gpu['scan_rts']
        del map_gpu['scan_seek_idx']
        del map_gpu['scan_im']
        del map_gpu['scan_mz']
        del map_gpu['scan_height']
        del map_gpu
        gc.collect()
        torch.cuda.empty_cache()
    except:
        pass


def convert_numba_to_tensor(x):
    x = cp.asarray(x).toDlpack()
    x = torch.from_dlpack(x)
    return x


def create_cuda_zeros(shape, dtype=torch.float32):
    x = torch.zeros(shape, dtype=dtype, device=param_g.device)
    x = cuda.as_cuda_array(x)
    return x


def get_diann_info(path_ws):
    if not param_g.is_compare_mode:
        return

    df_diann = pd.read_csv(path_ws / 'diann' / 'report.tsv', sep='\t')
    df_diann = df_diann[df_diann['Q.Value'] < 0.01]
    df_diann['pr_id'] = (df_diann['Modified.Sequence'] +
                         df_diann['Precursor.Charge'].astype(str))

    # rt
    rt_tol_diann = (df_diann['RT'] - df_diann['Predicted.RT']).abs().max() * 60.
    info = 'DIA-NN tol_rt: {:.2f}'.format(rt_tol_diann)
    logger.info(info)

    # im
    im_tol_diann = (df_diann['IM'] - df_diann['Predicted.IM']).abs().max()
    info = 'DIA-NN tol_im: {:.4f}'.format(im_tol_diann)
    logger.info(info)

    # ppm
    import re
    with open(path_ws / 'diann' / 'report.log.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.find('Recommended MS1 mass accuracy setting') > -1:
                pattern = r"\d+\.\d+"
                match = re.search(pattern, line)
                ppm_ms1 = match.group()
                logger.info('DIA-NN MS1 tol_ppm: {}'.format(ppm_ms1))
            if line.find('Optimised mass accuracy') > -1:
                pattern = r"\d+\.\d+"
                match = re.search(pattern, line)
                ppm_ms2 = match.group()
                logger.info('DIA-NN MS2 tol_ppm: {}'.format(ppm_ms2))
            if line.find('window radius') > -1:
                pw = int(line.split(' ')[-1])
                logger.info('DIA-NN peak width: {}'.format(pw))
            if line.find('Training neural networks') > -1:
                x = ' '.join(line.split(' ')[1:])
                logger.info('DIA-NN ' + x.strip())
            if line.find('Number of IDs at 0.01 FDR') > -1:
                t = line.split(' ')[0][1:-1]
                ids = line.split(' ')[-1]
                logger.info(f'DIA-NN time: {t}')
                logger.info(f'DIA-NN 1% FDR prs: {ids}')


def cal_acc_recall(path_ws, df_input,
                   diann_q_pr=None, diann_q_pro=None, diann_q_pg=None,
                   alpha_q_pr=None, alpha_q_pro=None, alpha_q_pg=None):
    if not param_g.is_compare_mode:
        return
    df_alpha = df_input.copy()

    if 'decoy' in df_alpha.columns:
        df_alpha = df_alpha[df_alpha['decoy'] == 0].reset_index(drop=True)

    # for diann pr
    df_diann = pd.read_csv(path_ws / 'diann' / 'report.tsv', sep='\t')
    if diann_q_pr is not None:
        df_diann_pr = df_diann[df_diann['Q.Value'] < diann_q_pr].copy()
    else:
        df_diann_pr = df_diann.copy()
    df_diann_pr['pr_id'] = (df_diann_pr['Modified.Sequence'] +
                            df_diann_pr['Precursor.Charge'].astype(str))
    pr_diann = set(df_diann_pr['pr_id'])

    # for beta pr
    if alpha_q_pr is not None:
        df_alpha_pr = df_alpha[df_alpha.q_pr < alpha_q_pr].copy()
    else:
        df_alpha_pr = df_alpha.copy()

    # intersection
    df_cross_pr = df_alpha_pr[['pr_id', 'measure_rt']]
    df_cross_pr = df_cross_pr.merge(df_diann_pr, on='pr_id')
    rt_delta = (df_cross_pr.measure_rt - df_cross_pr.RT * 60.).abs()
    df_cross_pr = df_cross_pr[rt_delta < param_g.locus_rt_thre]
    pr_cross_pr = set(df_cross_pr.pr_id)

    # recall and acc on pr level
    pr_alpha = set(df_alpha_pr['pr_id'])
    pr_recall_2 = len(pr_cross_pr) / (len(pr_diann) + 1)
    pr_recall_1 = len(pr_diann & pr_alpha) / (len(pr_diann) + 1)
    pr_acc = len(pr_diann & pr_alpha) / (len(pr_alpha) + 1)
    pr_gain = (len(pr_alpha) - len(pr_diann)) / (len(pr_diann) + 1)
    info = 'Df: {}, ' \
           'Prs: {}, ' \
           'Pr_gain: {:.2f}, ' \
           'pr_acc: {:.3f}, pr_recall_1: {:.3f}, pr_recall_2: {:.3f}'.format(
        len(df_alpha),
        len(pr_alpha),
        pr_gain,
        pr_acc, pr_recall_1, pr_recall_2,
    )

    # pro and pg
    if diann_q_pro or alpha_q_pro or diann_q_pg or alpha_q_pg:
        # protein
        df_diann_pro = df_diann[(df_diann['Protein.Q.Value'] < diann_q_pro) &
                                (df_diann['Proteotypic'] == 1)]
        df_alpha_pro = df_alpha[(df_alpha['q_pro'] < alpha_q_pro) &
                                (df_alpha['proteotypic'] == 1)].copy()

        pro_diann = set(df_diann_pro['Protein.Ids'])
        pro_alpha = set(df_alpha_pro['protein_id'])
        pro_recall = len(pro_diann & pro_alpha) / (len(pro_diann) + 1)
        pro_acc = len(pro_diann & pro_alpha) / (len(pro_alpha) + 1)
        pro_gain = (len(pro_alpha) - len(pro_diann)) / (len(pro_diann) + 1)

        # protein group
        df_diann_pg = df_diann[(df_diann['PG.Q.Value'] < diann_q_pg)]
        df_alpha_pg = df_alpha[(df_alpha['q_pg'] < alpha_q_pg)]

        pg_diann = set(df_diann_pg['Protein.Group']) # raw
        pg_alpha = set(df_alpha_pg['protein_group'])
        pg_recall = len(pg_diann & pg_alpha) / (len(pg_diann) + 1)
        pg_acc = len(pg_diann & pg_alpha) / (len(pg_alpha) + 1)
        pg_gain = (len(pg_alpha) - len(pg_diann)) / (len(pg_diann) + 1)

        info = 'Prs: {}, ' \
               'Pr_gain: {:.2f}, ' \
               'pr_acc: {:.3f}, pr_recall_1: {:.3f}, pr_recall_2: {:.3f}, ' \
               'Pro_num: {}, ' \
               'Pro_gain: {:.2f}, ' \
               'pro_acc: {:.2f}, pro_recall: {:.2f}, ' \
               'Pg_num: {}, ' \
               'pg_gain: {:.2f}, ' \
               'pg_acc: {:.2f}, pg_recall: {:.2f}'.format(
            len(pr_alpha),
            pr_gain,
            pr_acc, pr_recall_1, pr_recall_2,
            len(pro_alpha),
            pro_gain, pro_acc, pro_recall,
            len(pg_alpha),
            pg_gain, pg_acc, pg_recall
        )
    logger.info(info)


def save_as_pkl(df, fname):
    if param_g.is_save_pkl:
        df.to_pickle(param_g.dir_out / fname)


def save_as_tsv(df, fname):
    if param_g.is_save_final:
        df.to_csv(param_g.dir_out / fname, sep='\t', index=False)


@jit(nopython=True, nogil=True, parallel=True)
def cal_group_rank(x, group_size_cumsum):
    item_num = len(x)
    result = np.zeros(item_num)
    rank = np.zeros(item_num, dtype=np.int8)
    for i in prange(len(group_size_cumsum) - 1):
        start = group_size_cumsum[i]
        end = group_size_cumsum[i + 1]
        xx = x[start: end]
        idx = np.argsort(xx)[::-1]
        result[start: end] = idx + start
        rank[start: end] = np.arange(end - start) + 1
    return result, rank


def push_all_zeros_back(a):
    # Based on http://stackoverflow.com/a/42859463/3293881
    valid_mask = a != 0
    flipped_mask = valid_mask.sum(1, keepdims=1) > np.arange(a.shape[1] - 1, -1,
                                                             -1)
    flipped_mask = flipped_mask[:, ::-1]
    a[flipped_mask] = a[valid_mask]
    a[~flipped_mask] = 0
    return a


def cal_sa_by_np(x, y):
    '''
    x/y has to be two-dimentions
    '''
    norm_x = np.linalg.norm(x, axis=1)
    norm_y = np.linalg.norm(y, axis=1)
    norm_xy = norm_x * norm_y

    xy_sum = (x * y).sum(axis=1)
    sa = xy_sum / (norm_xy + 1e-7)
    sa = 1 - 2 * np.arccos(sa) / np.pi

    return sa


def convert_cols_to_diann(df):
    df = df.rename(columns={'protein_group': 'Protein.Group',
                            'protein_id': 'Protein.Ids',
                            'protein_name': 'Protein.Names',
                            'quant_pg': 'PG.Quantity',
                            'pr_id': 'Precursor.Id',
                            'pr_charge': 'Precursor.Charge',
                            'q_pr': 'Q.Value',
                            'q_pro': 'Protein.Q.Value',
                            'q_pg': 'PG.Q.Value',
                            'proteotypic': 'Proteotypic',
                            'quant_pr': 'Precursor.Quantity',
                            'measure_rt': 'RT',
                            'cscore_pr': 'CScore',
                            'cscore_pg': 'CScore.PG',
                            'measure_im': 'IM'})
    df = df[['Protein.Group', 'Protein.Ids', 'Protein.Names', 'PG.Quantity',
             'Precursor.Id', 'Precursor.Charge', 'Q.Value', 'Protein.Q.Value',
             'PG.Q.Value', 'Proteotypic', 'Precursor.Quantity', 'RT',
             'CScore', 'IM', 'CScore.PG'
             ]]

    return df


def get_args():
    import beta_dia
    name = f"Beta-DIA {beta_dia.__version__}"
    print(' ' * 9, "*" * (len(name) + 4))
    print(' ' * 9, f"* {name} *")
    print(' ' * 9, "*" * (len(name) + 4))

    parser = argparse.ArgumentParser('Beta-DIA for diaPASEF analysis')
    parser.add_argument(
        '-ws', '--ws', required=True,
        help='specify the folder that is .d or contains .d files.'
    )
    parser.add_argument(
        '-lib', '--lib', required=True,
        help='specify the absolute path of a .speclib spectra library.'
    )
    parser.add_argument(
        '-out_name', '--out_name', type=str, default='beta_dia',
        help='specify the folder name of outputs.'
    )
    args = parser.parse_args()
    return Path(args.ws), Path(args.lib), args.out_name


def init_multi_ws(ws):
    # GPU memory has to be larger than 10G!
    total_memory = torch.cuda.get_device_properties(0).total_memory
    if total_memory / 1024 ** 3 < 10:
        print('GPU memory is less than 10G. Beta-DIA may crash!')

    multi_ws = []
    if ws.suffix == '.d':
        multi_ws.append(ws)
    else:
        for ws_i in ws.rglob('*.d'):
            if ws_i.is_dir():
                multi_ws.append(ws_i)
    param_g.multi_ws = multi_ws
    param_g.file_num = len(param_g.multi_ws)

    i = 'The specified ws does not end with .d or does not contain a .d folder.'
    if param_g.file_num == 0:
        print(i)


def init_single_ws(ws_i, total, ws, out_name, dir_lib, entry_num):
    param_g.ws = ws
    param_g.dir_out = (ws / out_name)
    param_g.dir_out.mkdir(exist_ok=True)
    Logger.set_logger(param_g.dir_out, is_time_name=param_g.is_time_log)
    logger.info(f'====================={ws_i+1}/{total}=====================')
    logger.info('.d: ' + str(ws.name))
    logger.info('Lib: ' + Path(dir_lib).name)
    logger.info(f'Lib prs: {entry_num}')
