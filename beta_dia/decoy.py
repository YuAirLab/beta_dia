import math
import operator

import numpy as np
from numba import cuda

from beta_dia import param_g
from beta_dia import utils
from beta_dia.log import Logger

try:
    profile
except:
    profile = lambda x: x

logger = Logger.get_logger()

@cuda.jit(device=True)
def sj_gpu_sum(array):
    result = 0
    for i in array:
        result += i
    return result


@cuda.jit
def gpu_cal_fg_mz(n,
                  fg_num,
                  mass_v,
                  seq_len_cumsum_v,
                  fg_type_m,
                  fg_len_m,
                  fg_charge_m,
                  result_fg_mz):
    '''
    Calculate the fragment ion m/z values of decoys.
    Each thread is for an ion of a pr.
    '''
    thread_idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if thread_idx >= n:
        return

    k = thread_idx // fg_num
    fg_idx = thread_idx % fg_num

    # seq mass
    start = seq_len_cumsum_v[k]
    end = seq_len_cumsum_v[k + 1]
    pep_mass_v = mass_v[start: end]

    # fg
    fg_type = fg_type_m[k, fg_idx]
    fg_len = fg_len_m[k, fg_idx]
    fg_charge = fg_charge_m[k, fg_idx]
    mass_proton = 1.007276466771
    mass_h2o = 18.0105650638

    if fg_type == 2:  # 'y'
        mass_v = pep_mass_v[-fg_len:]
        mass = sj_gpu_sum(mass_v) - (fg_len - 1) * mass_h2o
        mz = (mass + fg_charge * mass_proton) / fg_charge
        result_fg_mz[k, fg_idx] = mz
    elif fg_type == 1:  # 'b'
        mass_v = pep_mass_v[:fg_len]
        mass = sj_gpu_sum(mass_v) - fg_len * mass_h2o
        mz = (mass + fg_charge * mass_proton) / fg_charge
        result_fg_mz[k, fg_idx] = mz


def convert_seq_to_mass(simple_seq):
    seq_len = simple_seq.str.len()
    seq_len_cumsum = np.concatenate([[0], np.cumsum(seq_len)])
    s = simple_seq.str.cat()
    s = list(s)

    f = operator.itemgetter(*s)
    mass = f(param_g.g_aa_to_mass)

    return mass, seq_len_cumsum


def cal_fg_mz_iso(df):
    mass_neutron = 1.0033548378

    cols_center = ['fg_mz_' + str(i) for i in range(param_g.fg_num)]
    fg_mz_m = df[cols_center].values

    cols_anno = ['fg_anno_' + str(i) for i in range(param_g.fg_num)]
    fg_anno_m = df[cols_anno].values
    fg_charge_m = fg_anno_m % 10

    fg_mz_left = (fg_mz_m * fg_charge_m - mass_neutron) / fg_charge_m
    fg_mz_left[fg_mz_m <= 0.] = 0.
    fg_mz_left = fg_mz_left.astype(np.float32)
    cols_left = ['fg_mz_left_' + str(i) for i in range(param_g.fg_num)]
    df[cols_left] = fg_mz_left

    fg_mz_1H = (fg_mz_m * fg_charge_m + mass_neutron) / fg_charge_m
    fg_mz_1H[fg_mz_m <= 0.] = 0.
    fg_mz_1H = fg_mz_1H.astype(np.float32)
    cols_1H = ['fg_mz_1H_' + str(i) for i in range(param_g.fg_num)]
    df[cols_1H] = fg_mz_1H

    fg_mz_2H = (fg_mz_m * fg_charge_m + 2 * mass_neutron) / fg_charge_m
    fg_mz_2H[fg_mz_m <= 0.] = 0.
    fg_mz_2H = fg_mz_2H.astype(np.float32)
    cols_2H = ['fg_mz_2H_' + str(i) for i in range(param_g.fg_num)]
    df[cols_2H] = fg_mz_2H

    return df


# @profile
def make_decoys(df_target, fg_num, method, value=1):
    '''
    Generate decoys. On fragment m/z values are changed. Pr_mz/Rt/IM are same.
    Args:
        df_target: only supports the modifications of C and M.
        fg_num:
        method: [reverse, mutate, shift]
        value: decoy is 1, shadow is 2

    Returns:
        df_decoy
    '''
    # mutate_dict
    mutate_dict = {}
    for old, new in zip('GAVLIFMPWSCTYHKRQEND', 'LLLVVLLLLTSSSSLLNDQE'):
        mutate_dict[old] = new

    # df_decoy
    if 'group_rank' in df_target.columns:
        df_decoy = df_target[df_target.group_rank == 1].copy()
    else:
        df_decoy = df_target.copy()

    if 'decoy' in df_target.columns:
        df_decoy = df_target[df_target.decoy == 0].copy()

    df_decoy = df_decoy.reset_index(drop=True)

    df_decoy['decoy'] = np.uint8(value)

    # change fg_mz
    if method == 'reverse':  # keep KR
        df_decoy['simple_seq'] = (
                df_decoy['simple_seq'].str[:-1].str[::-1] +
                df_decoy['simple_seq'].str[-1]
        )
    if method == 'shift':
        x = df_decoy['simple_seq']
        df_decoy['simple_seq'] = x.str[2:] + x.str[:2]
    elif method == 'mutate':
        second_bone_C = df_decoy.simple_seq.str[-2].str.upper()
        f = operator.itemgetter(*second_bone_C)
        second_bone_C = f(mutate_dict)

        second_bone_N = df_decoy.simple_seq.str[1].str.upper()
        f = operator.itemgetter(*second_bone_N)
        second_bone_N = f(mutate_dict)

        mutate = (df_decoy.simple_seq.str[0] + second_bone_N +
                  df_decoy.simple_seq.str[2:-2] + second_bone_C +
                  df_decoy.simple_seq.str[-1])
        df_decoy['simple_seq'] = mutate

    # update pr_id
    ModifiedPeptide = df_decoy['simple_seq'].replace(
        ['c', 'm'], ['C(UniMod:4)', 'M(UniMod:35)'], regex=True)
    df_decoy['pr_id'] = ModifiedPeptide + df_decoy['pr_charge'].astype(str)
    df_decoy['pr_root'] = df_target['pr_id']

    # drop duplicates and mismatch to target seqs
    df_decoy = df_decoy.drop_duplicates(subset='pr_id').reset_index(drop=True)
    bad_idx = df_decoy['pr_id'].isin(set(df_target['pr_id']))
    df_decoy = df_decoy.loc[~bad_idx].reset_index(drop=True)

    # calculating fg_mass in batches
    fg_mz_v = []

    for _, df_batch in df_decoy.groupby(df_decoy.index // 50000):
        # fg_anno: 2251 means y25_1
        cols_anno = ['fg_anno_' + str(i) for i in range(fg_num)]
        fg_anno = df_batch[cols_anno].values
        fg_type = (fg_anno // 1000).astype(np.int8)  # y-2, b-1, x-3
        fg_charge = (fg_anno % 10).astype(np.int8)
        fg_len = (fg_anno // 10 % 100).astype(np.int8)

        mass, seq_len_cumsum = convert_seq_to_mass(df_batch['simple_seq'])

        # by cuda
        mass = cuda.to_device(mass)
        seq_len_cumsum = cuda.to_device(seq_len_cumsum)
        fg_type = cuda.to_device(fg_type)
        fg_len = cuda.to_device(fg_len)
        fg_charge = cuda.to_device(fg_charge)
        result_fg_mz = utils.create_cuda_zeros((len(df_batch), fg_num))

        # kernel func
        n = result_fg_mz.shape[0] * result_fg_mz.shape[1]
        threads_per_block = 512
        blocks_per_grid = math.ceil(n / threads_per_block)
        gpu_cal_fg_mz[blocks_per_grid, threads_per_block](n,
                                                          fg_num,
                                                          mass,
                                                          seq_len_cumsum,
                                                          fg_type,
                                                          fg_len,
                                                          fg_charge,
                                                          result_fg_mz)
        cuda.synchronize()
        fg_mz_v.append(result_fg_mz.copy_to_host())
    fg_mz_v = np.vstack(fg_mz_v)
    cols_center = ['fg_mz_' + str(i) for i in range(fg_mz_v.shape[1])]
    df_decoy[cols_center] = fg_mz_v
    return df_decoy
