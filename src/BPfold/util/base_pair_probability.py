import os
import math

import numpy as np

from .misc import get_file_name


def read_BPPM_ribonanza(path, L):
    ret = np.zeros((L, L))
    with open(path) as f:
        for line in f.readlines():
            line = line.strip('\n ')
            if line:
                row, col, val = line.split(' ')
                row, col, val = int(row)-1, int(col)-1, float(val)
                ret[row, col] = ret[col, row] = val
    return ret


def read_BPPM(path, L=None):
    '''
        Generated BPPM, same format of Contrafold's
    '''
    triples = []
    maxLen = 0
    with open(path) as f:
        for line in f.readlines():
            line = line.strip('\n ')
            if line:
                idx, base, *connects = line.split(' ')
                for connect_p in connects:
                    if connect_p:
                        connect, p = connect_p.split(':')
                        row, col, val = int(idx)-1, int(connect)-1, float(p)
                        triples.append((row, col, val))
                        maxLen = max(maxLen, col+1)
                maxLen = max(maxLen, int(idx))
    if L is None:
        L = maxLen
    ret = np.zeros((L, L), dtype=np.float32)
    for row, col, val in triples:
        ret[row, col] = ret[col, row] = val
    return ret


def write_BPPM(dest, seq, mat):
    '''
        write mat (BPPM of seq) to dest, same format of Contrafold's
        mat: BPPM
    '''
    L = len(seq)
    assert L!=0 and L == len(mat) == len(mat[0])
    with open(dest, 'w') as fp:
        for idx, base in enumerate(seq):
            BPP_str = []
            for j in range(idx+1, L):
                if mat[idx][j]>0:
                    BPP_str.append(f'{j+1}:{mat[idx][j]:.6f}')
            fp.write(f'{idx+1} {base} {" ".join(BPP_str)}\n')


def CDP_BPPM(seq):
    def gaussian(x):
        return math.exp(-0.5*(x*x))
    def get_score(baseA, baseB):
        if {baseA, baseB} == {'A', 'U'}:
            return 2
        elif {baseA, baseB} == {'G', 'C'}:
            return 3
        elif {baseA, baseB} == {'G', 'U'}:
            return 0.8
        else:
            return 0
    L = len(seq)
    seq = seq.upper().replace('T', 'U')
    mat = np.zeros([L, L])
    for i in range(L):
        for j in range(L):
            for add in range(30):
                if i - add >= 0 and j + add <L:
                    score = get_score(seq[i-add], seq[j+add])
                    if score == 0:
                        break
                    else:
                        mat[i,j] = score * gaussian(add)
                else:
                    break
            if mat[i,j] > 0:
                for add in range(1, 30):
                    if i + add < L and j - add >= 0:
                        score = get_score(seq[i+add], seq[j-add])
                        if score == 0:
                            break
                        else:
                            mat[i,j] = score * gaussian(add)
                    else:
                        break
    return mat


def read_BPP_from_ps(ps_path, threshold=0.001):
    beginSeq = False
    beginBPP = False
    seq_lines = []
    triples = []
    with open(ps_path) as fp:
        for line in fp.readlines():
            if beginSeq:
                if line.startswith(') } def'):
                    beginSeq = False
                else:
                    seq_lines.append(line.strip(' \n\r\\'))
            else:
                if line.startswith('/sequence { ('):
                    beginSeq = True
            if beginBPP:
                if line.startswith('showpage'):
                    beginBPP = False
                else:
                    i, j, v, _ = line.split(' ')
                    i, j, v = int(i)-1, int(j)-1, float(v)
                    if v>=threshold:
                        triples.append((i, j, v))
            else:
                if line.startswith('%start of base pair prob'):
                    beginBPP = True
    seq = ''.join(seq_lines)
    L = len(seq)
    mat = np.zeros((L, L), dtype=float)
    for i, j, v in triples:
        mat[i][j] = mat[j][i] = v
    return seq, mat


def gen_BPPM(dest_path:str, mutated_seq:str, name:str, method:str='CDPfold', posteriors:float=0.001, prefix_dir:str='./BPPM_generator')->None:
    '''
    Generate BPPM of seq_path, write to dest_path

    Parameters
    ----------
    dest_path: str
        dest path: *.txt
    mutated_seq: str
        only contains AUGC
    name: str
    method: str
        'EternaFold', 'CDPfold', 'Contrafold', 'ViennaRNA'
    '''
    log_path = os.path.join(prefix_dir, '.gen_BPPM.log')
    # write seq and ss
    mut_seq_path = os.path.join(os.path.dirname(dest_path), name+'.fasta')
    write_fasta(mut_seq_path, [(name, mutated_seq)])

    method_low = method.lower()
    if method_low in {'eternafold', 'contrafold'}:
        paras = f'--params {prefix_dir}/EternaFold/parameters/EternaFoldParams.v1' if method_low == 'eternafold' else ''
        cmd_str = f'{prefix_dir}/EternaFold/src/contrafold predict "{mut_seq_path}" {paras} --posteriors {posteriors} "{dest_path}" >> {log_path} 2>&1'
        ret_code = os.system(cmd_str)
        if ret_code:
            raise Exception(f'Error encountered when using {method}')
    elif method_low == 'cdpfold':
        mat = CDP_BPPM(mutated_seq)
        write_BPPM(dest_path, mutated_seq, mat)
    elif method_low == 'viennarna':
        prefix = os.path.join(prefix_dir, method)
        cmd_str = f'{prefix}/bin/RNAfold -p --noPS "{mut_seq_path}" >> {log_path} 2>&1'
        ret_code = os.system(cmd_str)
        if ret_code:
            raise Exception(f'Error encountered when using {method}')
        ps_path = get_file_name(dest_seq_name)+'_dp.ps'
        cur_seq, mat = read_BPP_from_ps(ps_path)
        write_BPPM(dest_path, cur_seq, mat)
    else:
        raise Exception(f'Unknown method {method}')
