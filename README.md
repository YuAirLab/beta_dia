# Beta-DIA

Beta-DIA is a partially open-source, free-to-use Python software that provides comprehensive peptide/protein identification and accurate quantification results for diaPASEF data.

---
### Contents
**[Installation](#installation)**<br>
**[Usage](#usage)**<br>
**[Output](#output)**<br>

---
### Installation

We recommend using [Conda](https://www.anaconda.com/) to create a Python environment for using Beta-DIA, whether on Windows or Linux.

1. Create a Python environment with version 3.9.18.
    ```bash
    conda create -n beta_env python=3.9.18 "numpy<2.0.0"
    conda activate beta_env
    ```

2. Install the corresponding PyTorch and CuPy packages based on your CUDA version (which can be checked using the `nvidia-smi` command). Beta-DIA requires GPUs with over 10 GB of VRAM to run.
  - CUDA-12
    ```bash
    pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
    pip install cupy-cuda12x
    conda install cudatoolkit
    ```
  - CUDA-11
    ```bash
    pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu118
    pip install cupy-cuda11x
    conda install cudatoolkit
    ```

3. Install Beta-DIA
    ```bash
    pip install beta_dia
    ```
   
---
### Usage
```bash
beta_dia -lib "Absolute path of the spectral library" -ws "Absolute path of the .d folder or a folder containing multiple .d folders"
```
(Please note that the path needs to be enclosed in quotes if running on a Windows platform.)

- `-lib`<br>
This parameter is used to specify the absolute path of the spectral library.
Beta-DIA currently supports the spectral library with the suffix .speclib predicted by DIA-NN (v1.9 and v1.9.1) or .parquet produced by DIA-NN (>= v1.9). 
It supports oxygen modifications on methionine (M) but does not include modifications such as phosphorylation or acetylation. 
Refer to [this](https://github.com/vdemichev/DiaNN) for instructions on how to generate prediction spectral libraries using DIA-NN. 
Beta-DIA will develop its own predictor capable of forecasting the peptide retention time, ion mobility, and fragmentation pattern. 
It may also be compatible with other formats of spectral libraries based on requests.

- `-ws`<br>
This parameter is used to specify the .d folder or the folder containing multiple .d folders that need to be analyzed.

Other optional params are list below by entering `beta_dia -h`:
```
       ******************
       * Beta-DIA x.y.z *
       ******************
Usage: beta_dia -ws WS -lib LIB

optional arguments for users:
  -h, --help           Show this help message and exit.
  -ws WS               Specify the folder that is .d or contains .d files.
  -lib LIB             Specify the absolute path of a .speclib or .parquet spectra library.
  -out_name OUT_NAME   Specify the folder name of outputs. Default: beta_dia.
  -gpu_id GPU_ID       Specify the GPU-ID (e.g. 0, 1, 2) which will be used. Default: 0.
```

### Output
Beta-DIA will generate **`beta_dia/report_beta.log.txt`** and **`beta_dia/report.tsv`** in output folder. 
The report.tsv contains precursor and protein IDs, as well as plenty of associated information. 
Most column names are consistent with DIA-NN and are self-explanatory.

* **Protein.Group** - inferred proteins. Beta-DIA uses [IDPicker](https://pubs.acs.org/doi/abs/10.1021/pr070230d) algorithm to infer proteins. 
* **Protein.Ids** - all proteins matched to the precursor in the library.
* **Protein.Names** names (UniProt names) of the proteins in the Protein.Group.
* **PG.Quantity** quantity of the Protein.Group.
* **Precursor.Id** peptide seq + precursor charge.
* **Precursor.Charge** the charge of precursor.
* **Q.Value** run-specific precursor q-value.
* **Global.Q.Value** global precursor q-value.
* **Protein.Q.Value** run-specific q-value for the unique protein, that is protein identified with proteotypic (=specific to it) peptides.
* **PG.Q.Value** run-specific q-value for the protein group.
* **Global.PG.Q.Value** global q-value for the protein group.
* **Proteotypic** indicates the peptide is specific to a protein.
* **Precursor.Quantity** quantity of the precursor.
* **RT** the retention time of the precursor.
* **IM** the ion mobility of the precursor.
* **CScore** the final precursor score calculated by Beta-DIA after merging all sub-scores.
* **CScore.PG** the final protein group score calculated by Beta-DIA after merging multiple peptide scores.

---
## Troubleshooting
- Please create a GitHub issue and I will respond as soon as possible.
- Email to me: songjian2022@suda.edu.cn

---
## Citing Beta-DIA

Check out: [**Beta-DIA: Integrating learning-based and function-based feature scores to
optimize the proteome profiling of diaPASEF mass spectrometry data**](https://www.biorxiv.org/content/10.1101/2024.11.19.624419v1.full.pdf)

---
## Changelog

#### 0.1.0
  * FEAT: first commit on GitHub.
#### 0.1.1
  * FIX: loading .d even with missing or occupied frame.
#### 0.1.2
  * FEAT: support the selection of a GPU to be used.
#### 0.2.0
  * FEAT: polishing the prs with share fgs to optimize FDR control.
#### 0.2.1
  * FIX: polishing should be followed by qvalue calculation again. 
#### 0.3.0
  * FEAT: refactor code to speed up
#### 0.4.0
  * FEAT: use model_ys; select locus with x == 1 or deep == 1
#### 0.5.0
  * FEAT: speed up load_ms by numba index; add deep_big cut before fast fdr
#### 0.5.1
  * FIX: train mlp by data with good_big_cut of 1.5sigma; rank by deep_big
#### 0.5.2
  * FEAT: add param start_id
#### 0.5.3
  * FIX: calib m/z using np.float64; remove autocast; model_ys_fast is ok
#### 0.5.4
  * FEAT: two methods (ascending or descending) to polish prs; remove left > center
#### 0.6.0
  * FEAT: 5% method to polish dubilous prs, may less the ids
#### 0.7.0
  * FIX: polish dubious prs considering two more confident prs
  * FEAT: support .parquet lib
#### 0.8.0
  * FEAT: Support MBR as the default run mode