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
    <a href="https://bme.ustc.edu.cn/2023/0322/c28131a596069/page.htm"><strong>Peng Xiong</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=8eNm2GMAAAAJ"><strong>S. Kevin Zhou</strong></a>
  </p>
  <h2 align="center">Submitted</h2>
  <div align="center">
    <img src="bpm.png", width="800">
  </div>
  <p align="center">
    <a href="https://doi.org/10.1101/2024.10.22.619430">bioRxiv</a> | 
    <a href="https://www.biorxiv.org/content/10.1101/2024.10.22.619430.full.pdf">PDF</a> |
    <a href="https://github.com/heqin-zhu/BPfold">Code</a>
  </p>
</p>



<!-- vim-markdown-toc GFM -->

* [Introduction](#introduction)
* [Installation](#installation)
    * [Requirements](#requirements)
    * [Instructions](#instructions)
* [Reproduction](#reproduction)
* [Usage](#usage)
    * [BPfold motif library](#bpfold-motif-library)
    * [BPfold inference](#bpfold-inference)
* [Acknowledgement](#acknowledgement)
* [LICENSE](#license)
* [Citation](#citation)

<!-- vim-markdown-toc -->

## Introduction
![overview](overview.png)
RNA secondary structure plays essential roles in modeling RNA tertiary structure and further exploring the function of non-coding RNAs. Computational methods, especially deep learning methods, have demonstrated great potential and performance for RNA secondary structure prediction. However, the generalizability of deep learning models is a common unsolved issue in the situation of unseen out-of-distribution cases, which hinders the further improvement of accuracy and robustness of deep learning methods. Here we construct a base pair motif library which enumerates the complete space of locally adjacent three-neighbor base pair and records the thermodynamic energy of corresponding base pair motifs through de novo modeling of tertiary structures, and we further develop a deep learning approach for RNA secondary structure prediction, named BPfold, which employs hybrid transformer and convolutional neural network architecture and an elaborately designed base pair attention block to jointly learn representative features and relationship between RNA sequence and the energy map of base pair motif generated from the above motif library. Quantitative and qualitative experiments on sequence-wise datasets and family-wise datasets have demonstrated the great superiority of BPfold compared to other state-of-the-art approaches in both accuracy and generalizability. The significant performance of BPfold will greatly boost the development of deep learning methods for predicting RNA secondary structure and the further discovery of RNA structures and functionalities.


## Installation
### Requirements
- Linux system
- python3.6+
- anaconda

### Instructions
1. Clone this repo.
```shell
git clone git@github.com:heqin-zhu/BPfold.git
cd BPfold
```
2. Create and activate BPfold environment.
```shell
conda env create -f BPfold_environment.yaml
conda activate BPfold
```
3. Download [model_predict.tar.gz](https://github.com/heqin-zhu/BPfold/releases/download/v0.1/model_predict.tar.gz) in [releases](https://github.com/heqin-zhu/BPfold/releases) and decompress it.
```shell
wget https://github.com/heqin-zhu/BPfold/releases/download/v0.1/model_predict.tar.gz
tar -xzf model_predict.tar.gz -C paras
```
4. Download datasets [BPfold_data.tar.gz](https://github.com/heqin-zhu/BPfold/releases/download/v0.1/BPfold_data.tar.gz) in [releases](https://github.com/heqin-zhu/BPfold/releases) and decompress them.
```shell
wget https://github.com/heqin-zhu/BPfold/releases/download/v0.1/BPfold_data.tar.gz
mkdir BPfold_data && tar -xzf BPfold_data.tar.gz -C BPfold_data
```

## Reproduction
For reproduction of all the quantitative results, we provide the predicted secondary structures and model parameters of BPfold in experiments. You can **directly downalod** the predicted secondary structures by BPfold *or* **use BPfold** with trained parameters to predict these secondary structures, and then **evaluate** the predicted results.

**Directly download**
```shell
wget https://github.com/heqin-zhu/BPfold/releases/download/v0.1/BPfold_test_results.tar.gz
tar -xzf BPfold_test_results.tar.gz
```
**Use BPfold**
1. Download [BPfold_reproduce.tar.gz](https://github.com/heqin-zhu/BPfold/releases/download/v0.1/BPfold_reproduce.pth) in [releases](https://github.com/heqin-zhu/BPfold/releases).
```shell
wget https://github.com/heqin-zhu/BPfold/releases/download/v0.1/BPfold_reproduce.pth -P paras
```
2. Use BPfold to predict test sequences.
```shell
python3 -m BPfold.main --run_name BPfold_reproduce --batch_size 32 --phase test --data_dir BPfold_data --index_name data_index.yaml --use_BPE --normalize_energy >> BPfold_reproduce.log 2>&1
```

**Evaluate**
```shell
python3 -m BPfold.evaluate --data_dir BPfold_data --pred_dir BPfold_test_results
```

After running above commands for evaluation, you will see the following outputs:
```txt
Time used: 23s
[Summary] eval_BPfold_test_results.yaml
 Pred/Total num: [('PDB_test', 60, 60), ('Rfam14.5-14.10', 2048, 2048), ('archiveII', 3966, 3966), ('bpRNA', 1305, 1305)]
---------------------len>600---------------------
dataset         & num   & f1    & p     & r    \\
archiveII       & 55    & 0.377 & 0.547 & 0.313\\
--------------------len<=600---------------------
dataset         & num   & f1    & p     & r    \\
PDB_test        & 60    & 0.727 & 0.779 & 0.695\\
Rfam14.5-14.10  & 2048  & 0.791 & 0.777 & 0.824\\
archiveII       & 3911  & 0.926 & 0.927 & 0.929\\
bpRNA           & 1305  & 0.701 & 0.670 & 0.757\\
-----------------------all-----------------------
dataset         & num   & f1    & p     & r    \\
PDB_test        & 60    & 0.727 & 0.779 & 0.695\\
Rfam14.5-14.10  & 2048  & 0.791 & 0.777 & 0.824\\
archiveII       & 3966  & 0.918 & 0.922 & 0.921\\
bpRNA           & 1305  & 0.701 & 0.670 & 0.757\\
```

## Usage
### BPfold motif library
The base pair motif library is publicly available [here](https://raw.githubusercontent.com/heqin-zhu/BPfold/refs/heads/master/paras/key.energy), which contains the `motif`:`energy` pairs. The motif is represented as `sequence`\_`pairIdx`\_`pairIdx`\-`chainBreak` where pairIdx is 0-indexed, and the energy is a reference score of statistical and physical thermodynamic energy.
For instance, `CAAAAUG_0_6-3 -49.7835` represents motif `CAAAAUG` has a known pair `C-G` whose indexes are `0` and `6`, with chainBreak lying at position `3`.

### BPfold inference
Use BPfold to predict RNA secondary structures. The following are some examples.
```shell
python3 -m BPfold.predict --input examples/examples.fasta
python3 -m BPfold.predict --seq  UUAUCUCAUCAUGAGCGGUUUCUCUCACAAACCCGCCAACCGAGCCUAAAAGCCACGGUGGUCAGUUCCGCUAAAAGGAAUGAUGUGCCUUUUAUUAGGAAAAAGUGGAACCGCCUG   AGGCAGUGAUGAUGAAAAAAGAUUACCAUCAAACUUUGAGAGAUUCACAGCUCGUUGAUGCAUACUUCUUUAUAUUACCUGAGCCU
python3 -m BPfold.predict --input examples/URS0000D6831E_12908_1-117_CI0.931.bpseq --output results --save_type ct
```

Here are the outputs after running `python3 -m BPfold.predict --input examples/examples.fasta`:
```txt
>> Welcome to use "BPfold" for predicting RNA secondary structure!
Loading paras/model_predict/BPfold_1-6.pth
Loading paras/model_predict/BPfold_2-6.pth
Loading paras/model_predict/BPfold_3-6.pth
Loading paras/model_predict/BPfold_4-6.pth
Loading paras/model_predict/BPfold_5-6.pth
Loading paras/model_predict/BPfold_6-6.pth
[    1] saved in "BPfold_results/5s_Shigella-flexneri-3_CI0.980.bpseq", CI=0.980
CUGGCGGCAGUUGCGCGGUGGUCCCACCUGACCCCAUGCCGAACUCAGAAGUGAAACGCCGUAGCGCCGAUGGUAGUGUGGGGUCUCCCCAUGCGAGAGUAGGGAACUGCCAG
(((((((.....((((((((.....((((((.............))))..))....)))))).)).((.((....((((((((...))))))))....)).))...)))))))
[    2] saved in "BPfold_results/URS0000D6831E_12908_1-117_CI0.931.bpseq", CI=0.931
UUAUCUCAUCAUGAGCGGUUUCUCUCACAAACCCGCCAACCGAGCCUAAAAGCCACGGUGGUCAGUUCCGCUAAAAGGAAUGAUGUGCCUUUUAUUAGGAAAAAGUGGAACCGCCUG
......((((((..(.(((((.......))))))(((.((((.((......))..))))))).................))))))..(((......)))..................
Finished!
```

For more help information, please run command `python3 -m BPfold.predict -h` to see.

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
