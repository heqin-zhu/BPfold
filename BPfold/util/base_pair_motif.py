from itertools import product

import numpy as np


def gen_all_bpms(canonical_pairs=None, radius:int=3):
    '''
    Generate all outer and inner base pair motifs of six kinds of canonical pairs.

    Parameters
    ----------
    canonical_pairs: list
    list of canonical_pairs, default=['GC', 'CG', 'AU', 'UA', 'GU', 'UG']

    radius: int=3

    Returns
    -------
    yield: bpm
    '''
    assert radius>=2
    if canonical_pairs is None:
        canonical_pairs = ['GC', 'CG', 'AU', 'UA', 'GU', 'UG']
    for pair in canonical_pairs:
        yield from gen_base_pair_motif(pair, radius)


def gen_base_pair_motif(pair:str, radius:int=3):
    '''
    Generate  outer and inner base pair motifs of one input pair.

    Parameters
    ----------
    pair: str
    radius: int=3

    Returns
    -------
    yield: bpm
    '''

    assert radius>=2
    bases = 'AUGC'
    '''
    1 1
    1 4
    1 16
    1 64
    4 1
    4 4
    4 16
    4 64
    16 1
    16 4
    16 16
    16 64
    64 1
    64 4
    64 16
    3129
    '''
    # outer base pair motif: begin and end part of seq, including less than r neighbors  # 3129
    for l_len, r_len in product(range(4), range(4)):
        if l_len == r_len ==3:
            continue
        l_seqs = None
        r_seqs = None
        if l_len==0:
            l_seqs = ['']
        else:
            l_seqs = product(*[bases for i in range(l_len)])
        if r_len==0:
            r_seqs = ['']
        else:
            r_seqs = product(*[bases for i in range(r_len)])
        for l_seq, r_seq in product(l_seqs, r_seqs):
            li = [pair[0], *l_seq, *r_seq, pair[1], '_', '0', '_', str(l_len+r_len+1), '-', str(l_len)] # TODO, cb+1?
            yield ''.join(li)

    # inner chain-break base pair motif: dis > 2*r,  # 4096
    d = 2*radius
    for seq in product(*[bases for i in range(2*radius)]):
        li = [pair[0], *seq[:radius], *seq[radius:], pair[1], '_', '0', '_', str(2*radius+1), '-', str(radius)] # TODO< cb+1?
        yield ''.join(li)

    ## inner hairpin base pair motif: dis <= 2*r,  # 5440 = 4^3 ... + 4^6
    for loop_len in range(3, 2*radius+1): # start with 2, since no sharp loop: |i-j|>=2
        for seq in product(*[bases for i in range(loop_len)]):
            li = [pair[0], *seq, pair[1], '_', '0', '_', str(loop_len+1)]
            yield ''.join(li)


class BPM_energy:
    def __init__(self, path):
        self.energy_table = {}
        self.norm_energy_table = {}
        type_energy = {}
        with open(path) as fp:
            for line in fp.readlines():
                line = line.strip('\n ')
                if line and not line.startswith('#'):
                    BPM, energy = line.split()
                    self.energy_table[BPM] = float(energy)
                    # type energy
                    BPM_type = BPM[BPM.find('_'):]
                    if BPM_type in type_energy:
                        type_energy[BPM_type] = min(type_energy[BPM_type], self.energy_table[BPM])
                    else:
                        type_energy[BPM_type] = self.energy_table[BPM]
        for BPM in self.energy_table:
            BPM_type = BPM[BPM.find('_'):]
            self.norm_energy_table[BPM] = self.energy_table[BPM]/type_energy[BPM_type]


    def get_BPMotif(self, seq:str, i:int, j:int, r:int)->(str, str):
        '''
        Generate r-neighbor base pair motif for base pair (i, j) of seq
            ------i-----
                       |
            ------j-----

        Parameters
        ----------
        seq: str
            RNA sequence
        i: int
            index of base pair
        j: int
            index of base pair
        r: int
            r-neighbor

        Returns
        -------
        (bpm1, bpm2): (str, str)
            base pair motifs
        '''
        if i>j:
            i, j = j, i
        L = len(seq)
        # base pair motif 1: begin and end
        left = max(i-r, 0)
        right = min(j+r, L-1)
        bpm_seq1 = seq[j: right+1] + seq[left: i+1]
        bpm1 = bpm_seq1 + f'_0_{len(bpm_seq1)-1}-{right-j}'

        # base pair motif 2: middle
        bpm2 = None
        if j-i<=2*r: # hairpin loop
            bpm2 = seq[i:j+1] + f'_0_{j-i}'
        else:
            bpm2 = seq[i: i+r+1] + seq[j-r:j+1] + f'_0_{2*r+1}-{r}'
        return bpm1, bpm2

    def get_energy(self, seq:str, radius:int=3, normalize_energy=True, return_BPM=False):
        '''
        Generate energy map in shape of LxL according to input seq (L) and energy table

        Parameters
        ----------
        seq: str
            RNA sequence
        radius: int
            r-neighbor

        Returns
        -------
        energy_map: np.ndarray LxL
        '''
        seq = seq.upper()
        canonical_pairs = {'AU', 'UA', 'GC', 'CG', 'GU', 'UG'}
        L= len(seq)
        mat = np.zeros((2, L, L)) if normalize_energy else np.zeros((L, L))
        ret_BPM = {pair+'_0_1-0' for pair in canonical_pairs}
        for i in range(L):
            for j in range(L):
                pair = seq[i]+seq[j]
                if i==j or abs(j-i)<=3 or pair not in canonical_pairs:
                    pass
                elif i<j:
                    bpm1, bpm2 = self.get_BPMotif(seq, i, j, r=radius)
                    if normalize_energy:
                        mat[0][i][j] = self.norm_energy_table[bpm1]
                        mat[1][i][j] = self.norm_energy_table[bpm2]
                    else:
                        mat[i][j] = self.energy_table[bpm1] +self. energy_table[bpm2] - self.energy_table[pair+'_0_1-0']
                    ret_BPM.add(bpm1)
                    ret_BPM.add(bpm2)
                else:
                    if normalize_energy:
                        mat[0][i][j] = mat[0][j][i]
                        mat[1][i][j] = mat[1][j][i]
                    else:
                        mat[i][j] = mat[j][i]
        if return_BPM:
            return mat, ret_BPM
        else:
            return mat


if __name__ == '__main__':
    path = 'paras/key.energy'
    BPM_ene = BPM_energy(path)
    seq_lst = ['AU', 'AUU', 'UG', 'AA', 'AUGAC', 'AUGCGUUCCAU']
    for seq in seq_lst:
        mat = BPM_ene.get_energy(seq)
        norm_mat = BPM_ene.get_energy(seq, normalize_energy=True)
        print(seq)
        for arr in mat:
            print(arr.tolist())
        print('norm mat 0')
        for arr in norm_mat[0]:
            print(arr.tolist())
        print('norm mat 1')
        for arr in norm_mat[1]:
            print(arr.tolist())
