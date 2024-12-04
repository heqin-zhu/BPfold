import os
import argparse


from .util.misc import get_file_name
from .util.RNA_kit import read_SS, connects2dbn, write_fasta


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seq', type=str, help='RNA sequence')
    parser.add_argument('-i', '--input', type=str, help='Input file which contains RNA sequences, in format of FASTA (supporting multiple seqs), bpseq, ct, dbn, or any other txet file (Only the first line will be read as input sequence).')
    parser.add_argument('--print', action='store_true')
    parser.add_argument('--gen_fasta', action='store_true')

    args = parser.parse_args()

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
        write_fasta(f'{dir_name}.fasta', sorted(name_seq_pairs))
