<p align="center">

  <h1 align="center">Deep generalizable prediction of RNA secondary structure via base pair motif energy</h1>
  <p align="center">
    <a href="https://heqin-zhu.github.io/"><strong>Heqin Zhu</strong></a>
    ·
    <a href="https://fenghetan9.github.io/"><strong>Fenghe Tang</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=mlTXS0YAAAAJ"><strong>Quan Quan</strong></a>
    ·
    <a href="https://bme.ustc.edu.cn/2023/0918/c28132a612449/page.htm"><strong>Ke Chen</strong></a>
    ·
    <a href="https://bme.ustc.edu.cn/2023/0322/c28131a596069/page.htm"><strong>Peng Xiong*</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=8eNm2GMAAAAJ"><strong>S. Kevin Zhou*</strong></a>
  </p>
  <h2 align="center">Submitted</h2>
  <div align="center">
    <img src="base_pair_motif.png", width="800">
  </div>
  <p align="center">
    <a href="https://doi.org/10.1101/2024.10.22.619430">bioRxiv</a> | 
    <a href="https://www.biorxiv.org/content/10.1101/2024.10.22.619430.full.pdf">PDF</a> |
    <a href="https://github.com/heqin-zhu/BPfold">GitHub</a> |
    <a href="https://pypi.org/project/BPfold">PyPI</a>
    
  </p>
</p>



<!-- vim-markdown-toc GFM -->

* [Introduction](#introduction)
* [Installation](#installation)
    * [Requirements](#requirements)
    * [Instructions](#instructions)
* [Usage](#usage)
    * [BPfold motif library](#bpfold-motif-library)
    * [BPfold Prediction](#bpfold-prediction)
* [Reproduction](#reproduction)
* [Acknowledgement](#acknowledgement)
* [LICENSE](#license)
* [Citation](#citation)

<!-- vim-markdown-toc -->

## Introduction
![overview](overview.png)
Deep learning methods have demonstrated great performance for RNA secondary structure prediction. However, generalizability is a common unsolved issue on unseen out-of-distribution RNA families, which hinders further improvement of the accuracy and robustness of deep learning methods. Here we construct a base pair motif library that enumerates the complete space of locally adjacent three-neighbor base pair and records the thermodynamic energy of corresponding base pair motifs through _de novo_ modeling of tertiary structures, and we further develop a deep learning approach for RNA secondary structure prediction, named BPfold, which learns relationship between RNA sequence and the energy map of base pair motif. Experiments on sequence-wise and family-wise datasets have demonstrated the great superiority of BPfold compared to other state-of-the-art approaches in accuracy and generalizability. We hope this work contributes to integrating physical priors and deep learning methods for the further discovery of RNA structures and functionalities.


## Installation
### Requirements
- python3.8+
- anaconda

### Instructions
1. Create and activate BPfold environment.
```shell
conda env create -f BPfold_environment.yaml
conda activate BPfold
```
2. Install BPfold
```shell
pip3 install BPfold --index-url https://pypi.org/simple
```
3. Download [model_predict.tar.gz](https://github.com/heqin-zhu/BPfold/releases/latest/download/model_predict.tar.gz) in [releases](https://github.com/heqin-zhu/BPfold/releases) and decompress it.
```shell
wget https://github.com/heqin-zhu/BPfold/releases/latest/download/model_predict.tar.gz
tar -xzf model_predict.tar.gz
```
4. Optional: Download datasets [BPfold_data.tar.gz](https://github.com/heqin-zhu/BPfold/releases/latest/download/BPfold_data.tar.gz) in [releases](https://github.com/heqin-zhu/BPfold/releases) and decompress them.
```shell
wget https://github.com/heqin-zhu/BPfold/releases/latest/download/BPfold_data.tar.gz
tar -xzf BPfold_data.tar.gz 
```

## Usage
### BPfold motif library
The base pair motif library is publicly available in [releases](https://github.com/heqin-zhu/BPfold/releases), which contains the `motif`:`energy` pairs. The motif is represented as `sequence`\_`pairIdx`\_`pairIdx`\-`chainBreak` where pairIdx is 0-indexed, and the energy is a reference score of statistical and physical thermodynamic energy.
For instance, `CAAAAUG_0_6-3 -49.7835` represents motif `CAAAAUG` has a known pair `C-G` whose indexes are `0` and `6`, with chainBreak lying at position `3`.

>[!NOTE]
>The base pair motif library can be used as thermodynamic priors in other models.

### BPfold Prediction
Use BPfold to predict RNA secondary structures. The following are some examples. The `out_type` can be `csv`, `bpseq`, `ct` or `dbn`, which is defaultly set as `csv`.
```shell
BPfold --checkpoint_dir PATH_TO_CHECKPOINT_DIR --seq GGUAAAACAGCCUGU AGUAGGAUGUAUAUG --output BPfold_results
BPfold --checkpoint_dir PATH_TO_CHECKPOINT_DIR --input examples/examples.fasta --out_type csv # (multiple sequences are supported)
BPfold --checkpoint_dir PATH_TO_CHECKPOINT_DIR --input examples/URS0000D6831E_12908_1-117.bpseq # .bpseq, .ct, .dbn
```

<details>

<summary>Example of BPfold prediction</summary>

Here are the outputs after running `BPfold --checkpoint_dir model_predict --input examples/examples.fasta --out_type bpseq`:
```txt
>> Welcome to use "BPfold" for predicting RNA secondary structure!
Loading paras/model_predict/BPfold_1-6.pth
Loading paras/model_predict/BPfold_2-6.pth
Loading paras/model_predict/BPfold_3-6.pth
Loading paras/model_predict/BPfold_4-6.pth
Loading paras/model_predict/BPfold_5-6.pth
Loading paras/model_predict/BPfold_6-6.pth
[    1] saved in "BPfold_results/SS/5s_Shigella-flexneri-3.bpseq", CI=0.980
CUGGCGGCAGUUGCGCGGUGGUCCCACCUGACCCCAUGCCGAACUCAGAAGUGAAACGCCGUAGCGCCGAUGGUAGUGUGGGGUCUCCCCAUGCGAGAGUAGGGAACUGCCAG
(((((((.....((((((((.....((((((.............))))..))....)))))).)).((.((....((((((((...))))))))....)).))...)))))))
[    2] saved in "BPfold_results/SS/URS0000D6831E_12908_1-117.bpseq", CI=0.931
UUAUCUCAUCAUGAGCGGUUUCUCUCACAAACCCGCCAACCGAGCCUAAAAGCCACGGUGGUCAGUUCCGCUAAAAGGAAUGAUGUGCCUUUUAUUAGGAAAAAGUGGAACCGCCUG
......((((((..(.(((((.......))))))(((.((((.((......))..))))))).................))))))..(((......)))..................
Finished!
```

</details>

For more help information, please run command `BPfold -h` to see.

## Reproduction
For reproduction of all the quantitative results, we provide the predicted secondary structures and model parameters of BPfold in experiments. You can **directly downalod** the predicted secondary structures by BPfold *or* **use BPfold** with trained parameters to predict these secondary structures, and then **evaluate** the predicted results.

**Directly download**
```shell
wget https://github.com/heqin-zhu/BPfold/releases/latest/download/BPfold_test_results.tar.gz
tar -xzf BPfold_test_results.tar.gz
```
**Use BPfold**
1. Download [BPfold_reproduce.tar.gz](https://github.com/heqin-zhu/BPfold/releases/latest/download/BPfold_reproduce.pth) in [releases](https://github.com/heqin-zhu/BPfold/releases).
```shell
wget https://github.com/heqin-zhu/BPfold/releases/latest/download/model_reproduce.tar.gz
tar -xzf model_reproduce.tar.gz
```
2. Use BPfold to predict test sequences.

**Evaluate**
```shell
BPfold_eval --gt_dir BPfold_data --pred_dir BPfold_test_results
```

After running above commands for evaluation, you will see the following outputs:

<details>

<summary>Outputs of evaluating BPfold</summary>

```txt
Time used: 29s
[Summary] eval_BPfold_test_results.yaml
 Pred/Total num: [('PDB_test', 116, 116), ('Rfam12.3-14.10', 10791, 10791), ('archiveII', 3966, 3966), ('bpRNA', 1305, 1305), ('bpRNAnew', 5401, 5401)]
-------------------------len>600-------------------------
dataset         & num   & INF   & F1    & P     & R    \\
Rfam12.3-14.10  & 64    & 0.395 & 0.387 & 0.471 & 0.333\\
archiveII       & 55    & 0.352 & 0.311 & 0.580 & 0.242\\
------------------------len<=600-------------------------
dataset         & num   & INF   & F1    & P     & R    \\
PDB_test        & 116   & 0.817 & 0.814 & 0.840 & 0.801\\
Rfam12.3-14.10  & 10727 & 0.696 & 0.690 & 0.662 & 0.743\\
archiveII       & 3911  & 0.829 & 0.827 & 0.821 & 0.843\\
bpRNA           & 1305  & 0.670 & 0.658 & 0.599 & 0.770\\
bpRNAnew        & 5401  & 0.655 & 0.647 & 0.604 & 0.723\\
---------------------------all---------------------------
dataset         & num   & INF   & F1    & P     & R    \\
PDB_test        & 116   & 0.817 & 0.814 & 0.840 & 0.801\\
Rfam12.3-14.10  & 10791 & 0.694 & 0.689 & 0.660 & 0.741\\
archiveII       & 3966  & 0.823 & 0.820 & 0.818 & 0.834\\
bpRNA           & 1305  & 0.670 & 0.658 & 0.599 & 0.770\\
bpRNAnew        & 5401  & 0.655 & 0.647 & 0.604 & 0.723\\
```

</details>

## Acknowledgement
We appreciate the following open source projects:
- [UFold](https://github.com/uci-cbcl/UFold)
- [vigg_ribonanza](https://github.com/autosome-ru/vigg_ribonanza/)
- [e2efold](https://github.com/ml4bio/e2efold)

## LICENSE
[MIT LICENSE](LICENSE)

## Citation
If you use our code, please kindly consider to cite our paper:
```bibtex
@article {Zhu2024.10.22.619430,
    author = {Zhu, Heqin and Tang, Fenghe and Quan, Quan and Chen, Ke and Xiong, Peng and Zhou, S. Kevin},
    title = {Deep generalizable prediction of RNA secondary structure via base pair motif energy},
    elocation-id = {2024.10.22.619430},
    year = {2024},
    doi = {10.1101/2024.10.22.619430},
    publisher = {Cold Spring Harbor Laboratory},
    URL = {https://www.biorxiv.org/content/early/2024/10/25/2024.10.22.619430},
    eprint = {https://www.biorxiv.org/content/early/2024/10/25/2024.10.22.619430.full.pdf},
    journal = {bioRxiv}
}
```
