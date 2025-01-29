import os
import argparse

import tqdm

from .util.misc import get_file_name
from .util.RNA_kit import read_SS, connects2dbn, write_fasta


def get_dbn(dest, src, gt_dir=None):
    src = os.path.abspath(src)
    len_src = len(src)
    if dest is None:
        dest = f'dbn_{os.path.basename(src)}.txt'
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
            fp.write(f'>[{ct:>6d}] {name}\n')
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
    parser.add_argument('--print', action='store_true', help='print seq and SS')
    parser.add_argument('--gen_fasta', action='store_true')
    parser.add_argument('--get_dbn', action='store_true', help='Meanwhile specify --gt_dir, saved in "./dbn_dataname.txt"')
    parser.add_argument('--gt_dir', type=str)
    parser.add_argument('--show_examples', action='store_true', help='Show examples and exit.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.show_examples:
        print('python3 -m src.BPfold.kit --input example.bpseq --print')
        print('python3 -m src.BPfold.kit --input data_dir --gen_fasta')
        print('python3 -m src.BPfold.kit --input data_dir --get_dbn')
        print('python3 -m src.BPfold.kit --input data_dir --get_dbn --gt_dir gt_dir')
        exit()
    if args.print:
        seq, connects = read_SS(args.input)
        print(seq)
        print(connects2dbn(connects))
    if args.gen_fasta:
        dir_name = os.path.basename(args.input)
        name_seq_pairs = []
        for pre, ds, fs in os.walk(args.input):
            for f in fs:
                seq, _ = read_SS(os.path.join(pre, f))
                name = get_file_name(f)
                name_seq_pairs.append((name, seq))
        write_fasta(f'{dir_name}.fasta', name_seq_pairs)
    if args.get_dbn:
        get_dbn(dest=None, src=args.input, gt_dir=args.gt_dir)


if __name__ == '__main__':
    main()
