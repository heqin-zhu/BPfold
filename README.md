# BPfold

## Installation
```shell
conda env create -f BPfold_environment.yaml
```

## Run programs
### predict
```shell
python3 -m BPfold.predict --input examples/examples.fasta
python3 -m BPfold.predict --seq  UUAUCUCAUCAUGAGCGGUUUCUCUCACAAACCCGCCAACCGAGCCUAAAAGCCACGGUGGUCAGUUCCGCUAAAAGGAAUGAUGUGCCUUUUAUUAGGAAAAAGUGGAACCGCCUG   AGGCAGUGAUGAUGAAAAAAGAUUACCAUCAAACUUUGAGAGAUUCACAGCUCGUUGAUGCAUACUUCUUUAUAUUACCUGAGCCU
python3 -m BPfold.predict --input examples/bpRNA_RFAM_26347.bpseq --output results --save_type ct
```

### evaluate
```shell
python3 -m BPPfold.evaluate --pred_dir .runs/dim256/pred_epoch-296
```

### train
```shell
nohup python3 -m BPPfold.main --run_name .runs/dim256 --batch_size 32 -g 0 --phase train --dim 256 --lr 0.0005 --epoch 150 --nfolds 1 --save_freq 4 --config configs/config.yaml --index_name data_index.yaml --use_BPP > log256 2>&1 &
```

### test
```shell
nohup python3 -m BPPfold.main --run_name .runs/dim256 --batch_size 32 -g 0 --phase test  --dim 256 --ckpt_epoch_list 88 --use_BPP > logtest256 2>&1  &
nohup python3 -m BPPfold.main --run_name .runs/dim256 --batch_size 4 -g 0 --dim 256  --phase test --nfolds 1 --Lmax 1500 --use_BPP > logtest256 2>&1  &
```
