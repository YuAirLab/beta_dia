import time
import struct
from pathlib import Path
import multiprocessing as mp
import pyarrow.parquet as pq
import numpy as np
import pandas as pd

from beta_dia.log import Logger

try:
    profile
except:
    profile = lambda x: x

logger = Logger.get_logger()

def read_string(f):
    size = f.read(4)
    size = struct.unpack('<i', size)[0]
    if size:
        str = f.read(size)
    else:
        str = ''
    return str


def read_string_data(data, idx):
    size, idx = read_int32_data(data, idx)
    str = data[idx: (idx + size)]
    idx += size
    return str, idx


def read_array_int(f):
    size = f.read(4)
    size = struct.unpack('<i', size)[0]
    if size:
        a = f.read(size * 4)
        a = struct.unpack(str(size) + 'i', a)
    else:
        a = None
    return a


def read_int8(f):
    x = f.read(1)
    x = struct.unpack('b', x)[0]
    return x


def read_int8_data(data, idx):
    x = data[idx: (idx + 1)]
    x = struct.unpack('b', x)[0]
    idx += 1
    return x, idx


def read_int32(f):
    x = f.read(4)
    x = struct.unpack('<i', x)[0]
    return x


def read_int32_data(data, idx):
    x = data[idx: (idx + 4)]
    x = struct.unpack('<i', x)[0]
    idx += 4
    return x, idx


def read_float64(f):
    x = f.read(8)
    x = struct.unpack('<d', x)[0]
    return x


def read_float32(f):
    x = f.read(4)
    x = struct.unpack('<f', x)[0]
    return x


def read_float32_data(data, idx):
    x = data[idx: (idx + 4)]
    x = struct.unpack('<f', x)[0]
    idx += 4
    return x, idx


def read_head(f):
    version = f.read(4)
    version = struct.unpack('<i', version)[0]

    gen_decoy = f.read(4)
    gen_decoy = struct.unpack('<i', gen_decoy)[0]

    gen_charge = f.read(4)
    gen_charge = struct.unpack('<i', gen_charge)[0]

    infer_proteotypicity = f.read(4)
    infer_proteotypicity = struct.unpack('<i', infer_proteotypicity)[0]
    return version, gen_decoy, gen_charge, infer_proteotypicity


def read_proteins(f):
    protein_num = read_int32(f)
    sp_v, id_v, name_v, gene_v, name_index_v, gene_index_v, precursors_v = [], [], [], [], [], [], []
    for i in range(protein_num):
        sp = read_int32(f)
        size = read_int32(f)
        id = read_string(f)
        name = read_string(f)
        gene = read_string(f)
        name_index = read_int32(f)
        gene_index = read_int32(f)
        precursors = [read_int32(f) for _ in range(size)]
        assert min(precursors) >= 0

        sp_v.append(sp)
        id_v.append(id)
        name_v.append(name)
        gene_v.append(gene)
        name_index_v.append(name_index)
        gene_index_v.append(gene_index)
        precursors_v.append(precursors)

    df = pd.DataFrame({'protein.sp': sp_v,
                       'protein.id': id_v,
                       'protein.name': name_v,
                       'protein.gene': gene_v,
                       'protein.name.index': name_index_v,
                       'protein.gene.index': gene_index_v,
                       'protein.precursors': precursors_v})

    return df


# @profile
def read_protein_ids(f):
    protein_ids_num = read_int32(f)
    ids_v, names_v, genes_v, names_indices_v, genes_indices_v, proteins_v = [], [], [], [], [], []
    precursors_v = []
    for i in range(protein_ids_num):
        size = read_int32(f)
        ids = read_string(f)
        names = read_string(f)
        genes = read_string(f)
        names_indices = read_array_int(f)
        genes_indices = read_array_int(f)
        precursors = read_array_int(f)
        proteins = [read_int32(f) for _ in range(size)]
        if size:
            assert min(proteins) >= 0
            ids_v.append(ids)
            names_v.append(names)
            genes_v.append(genes)
            names_indices_v.append(names_indices)
            genes_indices_v.append(genes_indices)
            precursors_v.append(precursors)
            proteins_v.append(proteins)

    df = pd.DataFrame({'protein.ids': ids_v,
                       'protein.ids.names': names_v,
                       'protein.ids.genes': genes_v,
                       'protein.ids.names.indices': names_indices_v,
                       'protein.ids.genes.indices': genes_indices_v,
                       'protein.ids.precursors': precursors_v,
                       'protein.ids.proteins': proteins_v})

    return df


def read_precursor_seq(f):
    size = read_int32(f)
    seq_v = [read_string(f) for _ in range(size)]
    df = pd.DataFrame({'seq': seq_v})
    return df


def read_name(f):
    size = read_int32(f)
    name_v = [read_string(f) for _ in range(size)]
    df = pd.DataFrame({'name': name_v})
    return df


def read_gene(f):
    size = read_int32(f)
    gene_v = [read_string(f) for _ in range(size)]
    df = pd.DataFrame({'gene': gene_v})
    return df


# @profile
def read_entry_worker(binary_data, block_positions, block_label, worker_i):
    start = block_positions[worker_i]
    end = block_positions[worker_i + 1]
    # adjust start and end
    if worker_i > 0:
        binary_data_right = binary_data[start:]
        start = start + binary_data_right.find(block_label) + len(block_label)
    if worker_i != (len(block_positions) - 2):
        binary_data_right = binary_data[end:]
        end = end + binary_data_right.find(block_label) + len(block_label)
    block_data = binary_data[start: end]

    pr_index_v, pr_charge_v, pr_length_v = [], [], []
    pr_mz_v, pr_irt_v, pr_im_v = [], [], []
    pr_id_v, pr_proteotypic_v = [], []
    fg_num_v = []
    fg_mz_v, fg_height_v, fg_charge_v = [], [], []
    fg_type_v, fg_index_v, fg_loss_v = [], [], []

    block_idx = 0

    while block_idx < len(block_data):
        index, charge, length, mz, irt, srt, x1, im, x2 = \
            struct.unpack('3i6f', block_data[block_idx: (block_idx + 36)])
        block_idx += 36
        pr_index_v.append(index)
        pr_charge_v.append(charge)
        pr_length_v.append(length)
        pr_mz_v.append(mz)
        pr_irt_v.append(irt)
        pr_im_v.append(im)

        # x = struct.unpack('<6i', block_data[block_idx:(block_idx+24)])
        block_idx += 24

        fg_num, block_idx = read_int32_data(block_data, block_idx)
        fg_num_v.append(fg_num)

        fg = struct.unpack('2f4b' * fg_num,
                           block_data[block_idx:(block_idx + 12 * fg_num)])
        block_idx += (12 * fg_num)
        fg_mz_v.extend(fg[::6])
        fg_height_v.extend(fg[1::6])
        fg_charge_v.extend(fg[2::6])
        fg_type_v.extend(fg[3::6])
        fg_index_v.extend(fg[4::6])
        fg_loss_v.extend(fg[5::6])

        # dc, entry_flag, pr_mz, pid_index = struct.unpack(
        #     '<4i', block_data[block_idx:(block_idx + 16)]
        # )
        block_idx += 16
        # assert dc == 0

        # tmp = '<' + str(fg_num * 4) + 'f'
        # xx = struct.unpack(tmp, block_data[block_idx:(block_idx+fg_num*4*4)])
        block_idx += (fg_num+1)*4*4
        block_idx += (12 - fg_num) * 4
        pr_id, block_idx = read_string_data(block_data, block_idx)
        pr_id_v.append(pr_id)

        what = block_data[block_idx:(block_idx + 24)]
        block_idx += 24
        assert what == b'\x00\x00\x00\x00\x00\x00\x80?\x00\x00\x80?\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'

    pr_index_v = np.array(pr_index_v, dtype=np.int32)
    pr_charge_v = np.array(pr_charge_v, dtype=np.int8)
    pr_length_v = np.array(pr_length_v, dtype=np.int8)
    pr_mz_v = np.array(pr_mz_v, dtype=np.float32)
    pr_irt_v = np.array(pr_irt_v, dtype=np.float32)
    pr_im_v = np.array(pr_im_v, dtype=np.float32)
    fg_num_v = np.array(fg_num_v, dtype=np.int8)
    df = pd.DataFrame({'pr_id': pr_id_v,
                       'pr_index': pr_index_v,
                       'pr_charge': pr_charge_v,
                       'pr_len': pr_length_v,
                       'pr_mz': pr_mz_v,
                       'pred_irt': pr_irt_v,
                       'pred_iim': pr_im_v,
                       'fg_num': fg_num_v,
                       })
    assert sum(fg_loss_v) == 0, 'DIA-NN .speclib has fg_loss type!'
    assert len(df) == len(df.drop_duplicates(['pr_id', 'pr_index']))
    assert len(df) == df['pr_id'].nunique() == df['pr_index'].nunique()

    # unify to top-12，fg_anno code：y15_2 --> 2152
    fg_mz_v = np.array(fg_mz_v, dtype=np.float32)
    fg_height_v = np.array(fg_height_v, dtype=np.float32)
    fg_type_v = np.array(fg_type_v, dtype=np.int16)  # b-1, y-2
    fg_index_v = np.array(fg_index_v, dtype=np.int16) # from C-term
    fg_charge_v = np.array(fg_charge_v, dtype=np.int16)

    y_index = fg_type_v == 2 # index -> len
    pr_length_vv = np.repeat(pr_length_v, fg_num_v)
    fg_index_v[y_index] = pr_length_vv[y_index] - fg_index_v[y_index]

    assert fg_charge_v.max() < 10, 'fg_charge has to be less than 10!'
    fg_anno_v = fg_type_v * 1000 + fg_index_v * 10 + fg_charge_v

    mask = np.arange(fg_num_v.max()) < fg_num_v[:, None]
    fg_mz = np.zeros(mask.shape, dtype=np.float32)
    fg_mz[mask] = fg_mz_v
    fg_height = np.zeros(mask.shape, dtype=np.float32)
    fg_height[mask] = fg_height_v
    fg_anno = np.ones(mask.shape, dtype=np.int16) * 3011  # 3 is fg_loss
    fg_anno[mask] = fg_anno_v

    fg_num = fg_mz.shape[1]
    df[['fg_mz_' + str(i) for i in range(fg_num)]] = fg_mz
    df[['fg_height_' + str(i) for i in range(fg_num)]] = fg_height
    df[['fg_anno_' + str(i) for i in range(fg_num)]] = fg_anno

    # pr_id -- string
    df['pr_id'] = df['pr_id'].str.decode('utf-8')

    return df


# @profile
def read_diann_speclib(file_path, worker_num):
    with open(file_path, 'rb') as f:
        version, gen_decoys, gen_charges, infer_proteotypicity = read_head(f)

        name = read_string(f)
        fasta_name = read_string(f)

        df_protein = read_proteins(f)
        df_protein_ids = read_protein_ids(f)
        df_seq = read_precursor_seq(f)
        df_name = read_name(f)
        df_gene = read_gene(f)

        irt_min = read_float64(f)
        irt_max = read_float64(f)

        entry_num = read_int32(f)
        binary_data = f.read()

    # last index
    block_label = b'\x00\x00\x00\x00\x00\x00\x80?\x00\x00\x80?\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
    start = 0
    end = binary_data.rfind(block_label) + len(block_label)

    if worker_num > 1:
        block_positions = np.linspace(start, end, worker_num + 1).astype(int)
        pool = mp.Pool(worker_num)

        results = [pool.apply_async(
            read_entry_worker,
            args=(binary_data, block_positions, block_label, i)
        ) for i in range(worker_num)]
        results = [r.get() for r in results]  # get
        pool.close()
        pool.join()

        df_pr = pd.concat(results, ignore_index=True)
    else: # main thread
        block_positions = np.linspace(start, end, 2).astype(int)
        df_pr = read_entry_worker(
            binary_data, block_positions, block_label, 0
        )

    assert len(df_pr) == entry_num, 'Read .speclib ERROR!'

    df_protein['protein.id'] = df_protein['protein.id'].str.decode(
        'utf-8', 'ignore'
    )
    df_protein['protein.name'] = df_protein['protein.name'].str.decode(
        'utf-8', 'ignore'
    )
    df_protein['protein.gene'] = df_protein['protein.gene'].str.decode(
        'utf-8', 'ignore'
    )
    df_protein_ids['protein.ids'] = df_protein_ids['protein.ids'].str.decode(
        'utf-8', 'ignore'
    )
    df_protein_ids['protein.ids.names'] = df_protein_ids[
        'protein.ids.names'].str.decode('utf-8', 'ignore')
    df_protein_ids['protein.ids.genes'] = df_protein_ids[
        'protein.ids.genes'].str.decode('utf-8', 'ignore')

    return (
        version, name, fasta_name,
        df_protein, df_protein_ids, df_seq, df_name, df_gene,
        df_pr
    )


class Library():

    # @profile
    def __init__(self, dir_lib):
        dir_lib = Path(dir_lib)
        logger.info('Loading lib: ' + dir_lib.name)
        t0 = time.time()
        self.lib_type = dir_lib.suffix

        # parquet
        if self.lib_type == '.parquet':
            df = pq.read_pandas(dir_lib)
            df = df.to_pandas()
            assert (df['Fragment.Loss.Type'] == 'noloss').all()

            self.df_pr, self.df_map = self.construct_parquet_dfs(df)

        # speclib
        if self.lib_type == '.speclib':
            (
                version, name, fasta_name,
                df_protein, df_protein_ids, df_seq, df_name, df_gene,
                df_pr
            ) = read_diann_speclib(dir_lib, worker_num=1 if __debug__ else 8)
            assert version == -8, '.speclib is not from DIA-NN-1.9/1.9.1!'

            self.df_protein = df_protein
            self.df_protein_ids = df_protein_ids
            self.df_seq = df_seq
            self.df_name = df_name
            self.df_gene = df_gene
            self.df_pr = df_pr

        assert len(self.df_pr) == self.df_pr['pr_id'].nunique()

        t1 = time.time()
        logger.info(f'Loading lib finished: {(t1 - t0):.3}s')

    def __len__(self):
        return len(self.df_pr)

    @profile
    def construct_parquet_dfs(self, df):
        x = ['Precursor.Id', 'Modified.Sequence', 'Stripped.Sequence',
         'Precursor.Charge', 'Proteotypic', 'Decoy', 'N.Term', 'C.Term', 'RT',
         'IM', 'Q.Value', 'Peptidoform.Q.Value', 'PTM.Site.Confidence',
         'PG.Q.Value', 'Precursor.Mz', 'Product.Mz', 'Relative.Intensity',
         'Fragment.Type', 'Fragment.Charge', 'Fragment.Series.Number',
         'Fragment.Loss.Type', 'Exclude.From.Quant', 'Protein.Ids',
         'Protein.Group', 'Protein.Names', 'Genes']

        # info - pr
        fg_height_v = df['Relative.Intensity'].values.astype(np.float32)

        fg_height_max_idx = np.where(fg_height_v == fg_height_v.max())[0]
        pr_id_v = df.loc[fg_height_max_idx, 'Precursor.Id']
        good_idx = ~pr_id_v.duplicated() # for case: 1., 1., 0.8
        fg_height_max_idx = fg_height_max_idx[good_idx]

        pr_id_v = df.loc[fg_height_max_idx, 'Precursor.Id'].values
        pr_charge_v = df.loc[fg_height_max_idx, 'Precursor.Charge'].values
        pr_mz_v = df.loc[fg_height_max_idx, 'Precursor.Mz'].values
        pred_irt_v = df.loc[fg_height_max_idx, 'RT'].values
        pred_iim_v = df.loc[fg_height_max_idx, 'IM'].values
        pr_length_v = df.loc[fg_height_max_idx, 'Stripped.Sequence'].str.len().values

        # info - fg
        fg_num_v = np.diff(fg_height_max_idx)
        fg_num_v = np.append(fg_num_v, len(df) - fg_height_max_idx[-1])
        fg_mz_v = df['Product.Mz'].values.astype(np.float32)
        fg_type_v = np.where(df['Fragment.Type'] == 'b', 1, 2)
        fg_type_v = fg_type_v.astype(np.int16)  # b-1, y-2
        fg_index_v = df['Fragment.Series.Number'].values.astype(np.int16)
        fg_charge_v = df['Fragment.Charge'].values.astype(np.int16)
        assert fg_charge_v.max() < 10
        assert fg_index_v.max() < 100
        fg_anno_v = fg_type_v * 1000 + fg_index_v * 10 + fg_charge_v
        mask = np.arange(fg_num_v.max()) < fg_num_v[:, None]
        fg_mz = np.zeros(mask.shape, dtype=np.float32)
        fg_mz[mask] = fg_mz_v
        fg_height = np.zeros(mask.shape, dtype=np.float32)
        fg_height[mask] = fg_height_v
        fg_anno = np.ones(mask.shape, dtype=np.int16) * 3011  # 3 is fg_loss
        fg_anno[mask] = fg_anno_v

        # df_map
        protein_id_v = df.loc[fg_height_max_idx, 'Protein.Ids'].values
        protein_name_v = df.loc[fg_height_max_idx, 'Protein.Names'].values
        # pg_v = df.loc[fg_height_max_idx, 'Protein.Group'].values
        gene_v = df.loc[fg_height_max_idx, 'Genes'].values
        df_map = pd.DataFrame()
        df_map['protein_id'] = protein_id_v
        df_map['protein_name'] = protein_name_v
        df_map['gene'] = gene_v

        # df_pr
        df_pr = pd.DataFrame()
        df_pr['pr_id'] = pr_id_v
        df_pr['pr_charge'] = pr_charge_v.astype(np.int8)
        df_pr['pr_mz'] = pr_mz_v.astype(np.float32)
        df_pr['pr_len'] = pr_length_v.astype(np.int8)
        df_pr['pred_irt'] = pred_irt_v.astype(np.float32)
        df_pr['pred_iim'] = pred_iim_v.astype(np.float32)
        df_pr['fg_num'] = fg_num_v.astype(np.int8)
        df_pr['pr_index'] = df_pr.index.values.astype(np.int32)
        fg_num = fg_mz.shape[1]
        df_pr[['fg_mz_' + str(i) for i in range(fg_num)]] = fg_mz
        df_pr[['fg_height_' + str(i) for i in range(fg_num)]] = fg_height
        df_pr[['fg_anno_' + str(i) for i in range(fg_num)]] = fg_anno

        assert len(df_pr) == len(df_map)

        return df_pr, df_map

    # @profile
    def polish_lib_by_swath(self, swath, ws_diann=None):
        df_lib = self.df_pr

        # for debug
        if ws_diann is not None:
            df_diann = pd.read_csv(ws_diann / 'diann' / 'report.tsv', sep='\t')
            df_diann = df_diann[df_diann['Q.Value'] < 0.01]
            df_diann = df_diann[['Modified.Sequence', 'Precursor.Charge',
                                 'RT', 'IM', 'Precursor.Quantity']]
            df_diann['diann_rt'] = df_diann['RT'] * 60.
            df_diann['diann_im'] = df_diann['IM']
            df_diann['diann_pr_quant'] = df_diann['Precursor.Quantity']
            df_diann['pr_id'] = df_diann['Modified.Sequence'] + df_diann[
                'Precursor.Charge'].astype(str)
            df = pd.merge(df_lib, df_diann, on='pr_id')
            df = df.reset_index(drop=True)
            del df['Modified.Sequence']
            del df['Precursor.Charge']
            del df['RT']
            del df['IM']
            del df['Precursor.Quantity']
            df_lib = df

        # screen prs by range of m/z
        pr_mz = df_lib['pr_mz'].values
        pr_mz_min, pr_mz_max = swath[0], swath[-1]
        good_idx = (pr_mz > pr_mz_min) & (pr_mz < pr_mz_max)
        df_lib = df_lib.iloc[good_idx].reset_index(drop=True)

        # drop duplicates
        df_lib = df_lib.drop_duplicates(subset='pr_id', ignore_index=True)
        assert len(df_lib) == df_lib.pr_id.nunique()

        # remove BJOUXZ
        df_lib['simple_seq'] = df_lib['pr_id'].str[:-1].replace(
            ['C\(UniMod:4\)', 'M\(UniMod:35\)'], ['c', 'm'], regex=True
        )
        bad_idx = df_lib['simple_seq'].str.contains('[BJOUXZ]', regex=True)
        df_lib = df_lib[~bad_idx].reset_index(drop=True)

        # fg_num >= 4
        df_lib = df_lib[df_lib.fg_num >= 4]
        df_lib = df_lib.reset_index(drop=True)

        # pred_im
        df_lib['pred_im'] = df_lib['pred_iim']

        # pr_mz_iso
        mass_neutron = 1.0033548378
        pr_mass = df_lib['pr_mz'] * df_lib['pr_charge']
        pr_mz_1H = (pr_mass + mass_neutron) / df_lib['pr_charge']
        pr_mz_2H = (pr_mass + 2 * mass_neutron) / df_lib['pr_charge']
        pr_mz_left = (pr_mass - mass_neutron) / df_lib['pr_charge']
        df_lib['pr_mz_1H'] = pr_mz_1H.astype(np.float32)
        df_lib['pr_mz_2H'] = pr_mz_2H.astype(np.float32)
        df_lib['pr_mz_left'] = pr_mz_left.astype(np.float32)

        # assign swath_id
        swath_id = np.digitize(df_lib['pr_mz'].values, swath)
        df_lib['swath_id'] = swath_id.astype(np.int8)
        idx = np.argsort(swath_id)
        df_lib = df_lib.iloc[idx].reset_index(drop=True)

        # decoy
        df_lib['decoy'] = np.uint8(0)

        # shuffle
        np.random.seed(1)
        df_lib = df_lib.sample(frac=1, random_state=1).reset_index(drop=True)

        logger.info('Polishing spectral library finished.')

        return df_lib

    def polish_lib_by_targets(self, pr_targets):
        self.df_pr = self.df_pr[self.df_pr['pr_id'].isin(pr_targets)]

    def assign_proteins(self, df):
        # find corresponding protein and name by pr_index
        if self.lib_type == '.parquet':
            df_map = self.df_map
            pr_index_q = df['pr_index'].values
            result_protein_id = df_map.loc[pr_index_q, 'protein_id']
            result_protein_name = df_map.loc[pr_index_q, 'protein_name']

        if self.lib_type == '.speclib':
            df_map = self.df_protein_ids

            pr_num = df_map['protein.ids.precursors'].apply(len)
            protein_indices = np.repeat(df_map.index.values, pr_num)
            pr_indices = df_map['protein.ids.precursors'].explode()
            pr_indices = pr_indices.values
            assert len(protein_indices) == len(pr_indices)
            assert len(pr_indices) == len(np.unique(pr_indices))

            pr_to_prot = pd.Series(protein_indices, index=pr_indices)
            query_idx = df['pr_index'].values
            protein_rows = pr_to_prot[query_idx].values
            result_protein_id = df_map.loc[protein_rows, 'protein.ids']
            result_protein_name = df_map.loc[protein_rows, 'protein.ids.names']

        df['protein_id'] = result_protein_id.values
        df['protein_name'] = result_protein_name.values

        df['proteotypic'] = df['protein_id'].str.count(';') + 1
        df.loc[df['proteotypic'] != 1, 'proteotypic'] = 0

        # add DECOY
        if 'decoy' in df.columns:
            decoy_idx = df['decoy'] == 1
            df.loc[decoy_idx, 'protein_id'] = 'DECOY_' + df.loc[
                decoy_idx, 'protein_id']
            df.loc[decoy_idx, 'protein_id'] = df.loc[
                decoy_idx, 'protein_id'].replace(';', ';DECOY_', regex=True)

            df.loc[decoy_idx, 'protein_name'] = 'DECOY_' + df.loc[
                decoy_idx, 'protein_name']
            df.loc[decoy_idx, 'protein_name'] = df.loc[
                decoy_idx, 'protein_name'].replace(';', ';DECOY_', regex=True)

        return df
