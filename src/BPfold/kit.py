import os
import argparse

import tqdm
import numpy as np
import pandas as pd

from .util.base_pair_motif import BPM_energy
from .util.misc import get_file_name, str_localtime
from .util.RNA_kit import read_SS, connects2dbn, mut_seq, read_fasta, write_fasta 


def get_dbn(dest, src, gt_dir=None):
    src = os.path.abspath(src)
    len_src = len(src)
    sufs = ['bpseq', 'ct', 'fasta']
    sufs += [i.upper() for i in sufs]
    paths = [os.path.abspath(os.path.join(pre, f)) for pre, ds, fs in os.walk(src) for f in fs if any([f.endswith(suf) for suf in sufs])]
    with open(dest, 'w') as fp:
        ct = 0
        for src_path in tqdm.tqdm(paths):
            ct+=1
            name = get_file_name(src_path)
            seq, connects= read_SS(src_path)
            dbn= connects2dbn(connects)
            fp.write(f'>{name}\n')
            fp.write(f'{seq}\n')
            fp.write(f'{dbn}\n')
            if gt_dir:
                gt_path = os.path.join(gt_dir, src_path[len_src:].strip(os.path.sep))
                seq_gt, connects_gt = read_SS(gt_path)
                dbn_gt = connects2dbn(connects_gt)
                assert seq_gt.upper()==seq.upper(), f"{src_path}, {gt_path}\nseq1: {seq}\nseq2: {seq_gt}\n"
                fp.write(f'{dbn_gt} native\n')
    print(f'Result saved in {dest}.')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str)
    parser.add_argument('-o', '--output', type=str, default='BPfold_kit_results')
    parser.add_argument('--print', action='store_true', help='print seq and SS')
    parser.add_argument('--get_fasta', action='store_true')
    parser.add_argument('--get_dbn', action='store_true', help='Meanwhile specify --gt_dir, saved in "./dbn_dataname.txt"')
    parser.add_argument('--get_matrix', type=str, choices=['energy', 'probability', ''], default='', help='input seq in shape of L, if energy: output nomralized outer BPM and inner BPM in shape of (2, L, L); if probability: output reference probability converted from energy in shape of (1, L, L).')
    parser.add_argument('--gt_dir', type=str)
    parser.add_argument('--name_col', type=str, default='target_id')
    parser.add_argument('--seq_col', type=str, default='sequence')
    parser.add_argument('--show_examples', action='store_true', help='Show examples and exit.')
    args = parser.parse_args()
    return args


def get_matrix(dest, name_seq_pairs, tag='energy'):
    assert tag in {'energy', 'probability'}, f'Unknown tag {tag}'
    dest = os.path.join(dest, f'{tag}_matrix')
    os.makedirs(dest, exist_ok=True)
    BPM_ene = BPM_energy()
    for name, seq in name_seq_pairs:
        seq = seq.upper().replace('T', 'U')
        m_seq = mut_seq(seq)
        if tag == 'energy':
            mat = BPM_ene.get_energy(m_seq, normalize_energy=True)
        else:
            mat = BPM_ene.get_probability(m_seq)
        np.save(os.path.join(dest, name+'.npy'), mat)


def main():
    args = parse_args()
    if args.show_examples:
        print('BPfold_kit --input example.bpseq --print')
        print('BPfold_kit --input data_dir --get_fasta')
        print('BPfold_kit --input data.csv --get_fasta --name_col target_id --seq_col sequence')
        print('BPfold_kit --input data_dir --get_dbn')
        print('BPfold_kit --input data_dir --get_dbn --gt_dir gt_dir')
        print('BPfold_kit --input seq --get_matrix energy')
        exit()
    os.makedirs(args.output, exist_ok=True)
    if os.path.isdir(args.input):
        save_name = os.path.basename(args.input)
    else:
        save_name = get_file_name(args.input)
    dest_path = None
    if args.print:
        seq, connects = read_SS(args.input)
        dbn = connects2dbn(connects)
        print(seq)
        print(dbn)
        dest_path = os.path.join(args.output, f'{save_name}.dbn')
        with open(dest_path, 'w') as fp:
            fp.write(f'>{save_name}\n{seq}\n{dbn}')
    if args.get_fasta:
        if os.path.isdir(args.input):
            name_seq_pairs = []
            for pre, ds, fs in os.walk(args.input):
                for f in fs:
                    seq, _ = read_SS(os.path.join(pre, f))
                    name = get_file_name(f)
                    name_seq_pairs.append((name, seq))
        elif args.input.endswith('.csv'):
            df = pd.read_csv(args.input)
            name_seq_pairs = zip(df[args.name_col], df[args.seq_col])
        else:
            raise Exception(args.input)
        dest_path = os.path.join(args.output, f'{save_name}.fasta')
        write_fasta(dest_path, name_seq_pairs)
    if args.get_dbn:
        dest_path = os.path.join(args.output, f'dbn_{save_name}.txt')
        get_dbn(dest=dest_path, src=args.input, gt_dir=args.gt_dir)
    if args.get_matrix:
        dest_path = args.output
        if os.path.exists(args.input): # fasta file
            get_matrix(args.output, read_fasta(args.input), args.get_matrix)
        else:
            time_str = str_localtime()
            get_matrix(args.output, [(f'seq_{time_str}', args.input)], args.get_matrix) # seq

    if dest_path is not None and os.path.exists(dest_path):
        print(f'Saving results in "{dest_path}"')


if __name__ == '__main__':
    main()
