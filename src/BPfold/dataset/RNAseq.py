import os
import yaml
import random
from itertools import product

import numpy as np
import torch
import torch.utils.data as data
from sklearn.model_selection import KFold

from ..util.misc import get_file_name, str_localtime
from ..util.RNA_kit import read_SS, connects2mat, mut_seq
from ..util.postprocess import get_base_index
from ..util.base_pair_motif import BPM_energy
from ..util.base_pair_probability import read_BPPM, gen_BPPM


CANONICAL_PAIRS = {'AU', 'UA', 'GC', 'CG', 'GU', 'UG'}


class RNAseq_data(data.Dataset):
    def __init__(self, 
                 data_dir,
                 index_name='data_index.yaml',
                 phase='train',
                 Lmax=600,
                 Lmin=0,
                 fold=0,
                 nfolds=4,
                 seed=42,
                 cache_dir='.cache_data', 
                 mask_only=False,
                 method='CDPfold',
                 trainall=False,
                 predict_files=None,
                 training_set=None,
                 test_set=None,
                 use_BPE=True, 
                 use_BPP=True, 
                 normalize_energy=False,
                 verbose=False,
                 para_dir=None,
                 BPM_type='all',
                 *args,
                 **kargs,
                 ):
        # Set all input args as attributes
        self.__dict__.update(locals())
        self.use_BPE = use_BPE
        self.use_BPP = use_BPP
        self.BPM_type = BPM_type
        self.phase = phase.lower()
        self.verbose = verbose
        if self.phase == 'predict':
            self.cache_dir = os.path.join(cache_dir, method)
        else:
            self.cache_dir = os.path.join(data_dir, cache_dir, method)
        self.method = method
        self.data_dir = data_dir
        self.mask_only = mask_only
        if use_BPP:
            os.makedirs(self.cache_dir, exist_ok=True)

        # data filter
        index_file = os.path.join(data_dir, index_name)
        if self.phase == 'predict':
            self.file_list = predict_files
            self.Lmax = max([f['length'] for f in self.file_list]) if predict_files else float('inf')
        else:
            if self.phase in {'train', 'validate'}:
                with open(index_file) as f:
                    index_dic = yaml.load(f.read(), Loader=yaml.FullLoader)
                    all_files = index_dic['train'][:]
                    if 'validate' in index_dic:
                        all_files += index_dic['validate']
                    if trainall:
                        all_files += index_dic['test']
                    # train with specific datasets
                    if self.phase == 'train' and training_set:
                        training_set = set(training_set)
                        if 'PDB_test' in training_set:
                            training_set.update({'PDB_test-TS1', 'PDB_test-TS2', 'PDB_test-TS3'})
                        print(f'training sets: {training_set}')
                        all_files = [f for f in all_files if f['dataset'] in training_set]

                    all_files.sort(key=lambda dic: dic['path'])

                    # Kfold
                    if nfolds <= 1: # no kfold
                        if self.phase == 'validate':
                            self.file_list = index_dic['test'][:]
                        else: # train
                            self.file_list = all_files[:]
                            random.shuffle(self.file_list)
                    else: # kfold training
                        split = list(KFold(n_splits=nfolds, random_state=seed, shuffle=True).split(range(len(all_files))))[fold][0 if self.phase == 'train' else 1]
                        self.file_list = [all_files[i] for i in split]
                self.Lmax = Lmax
                # limit length
                self.file_list = [f for f in self.file_list if Lmin<=f['length']<=self.Lmax]

            elif self.phase in ['test']:
                with open(index_file) as f:
                    index_dic = yaml.load(f.read(), Loader=yaml.FullLoader)
                    self.file_list = index_dic['test'][:]
                    if self.phase == 'test' and test_set:
                        test_set = set(test_set)
                        if 'PDB_test' in test_set:
                            test_set.update({'PDB_test-TS1', 'PDB_test-TS2', 'PDB_test-TS3'})
                        print(f'test sets: {test_set}')
                        self.file_list = [f for f in self.file_list if f['dataset'] in test_set]
                self.Lmax = max([f['length'] for f in self.file_list])
            else:
                raise NotImplementedError
            print(f'phase={self.phase}, num={len(self.file_list)}, nfolds={nfolds}: {index_file}')
            print(f'use_BPP={use_BPP}, use_BPE={use_BPE}')
            for dic in self.file_list:
                dic['path'] = os.path.join(self.data_dir, dic['path'])

        if para_dir is None:
            para_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'paras')
        if self.use_BPE:
            self.normalize_energy = normalize_energy
            self.BPM_ene = BPM_energy(path=os.path.join(para_dir, 'key.energy'))
        self.base_index = get_base_index() # for matrix embed
        self.num_base = len(self.base_index)
        self.index_base = {v: k for k, v in self.base_index.items()}
        self.token_index = {k: v for k, v in self.base_index.items()}
        self.token_index.update({tok: len(self.base_index)+i for i, tok in enumerate(['START', 'END', 'EMPTY'])}) # for sequence embed
        self.noncanonical = [self.index_base[i]+self.index_base[j] not in CANONICAL_PAIRS for i, j in product(range(self.num_base), range(self.num_base))]
        self.noncanonical_flag = np.array(self.noncanonical, dtype=bool)
        self.to_device_keywords = {'input', 'input_seqmat', 'mask', 'forward_mask', 'BPPM', 'BPEM', 'seq_onehot', 'nc_map',}
        if self.phase !='predict':
            self.to_device_keywords.add('gt')
            print(self.token_index)


    def __len__(self):
        return len(self.file_list)

    def prepare_data(self, name, seq, connects=None):
        ret = {}
        L = len(seq)
        ret['ori_seq'] = seq.upper() # AUGC and others
        ret['seq'] = mut_seq(ret['ori_seq'].replace('T', 'U'), connects) # unknown -> AUGC
        ret['length'] = L

        # mask: (Lmax+2)x(Lmax+2)
        mask = torch.zeros(self.Lmax + 2, self.Lmax+2, dtype=torch.bool)
        for row in range(1, L+1): # not including START and END
            mask[row, 1:L+1] = True
        # forward_mask: Lmax+2
        forward_mask = torch.zeros(self.Lmax + 2, dtype=torch.bool) # START, seq, END
        forward_mask[0:L+2] = True # including START and END
        ret['mask'] = mask
        ret['forward_mask'] = forward_mask

        if self.mask_only:
            return ret, None

        ## pad to uniform size=(Lmax+2) when batch-loading
        rside_pad = self.Lmax + 1 - L

        # seq_embed seq: Lmax+2  
        ret['input'] = self.seq_embed_sequence(ret['seq'])

        # seq_embed outer product 
        seqmat, seq_onehot = self.seq_embed_matrix(ret['seq'], return_onehot=True) # N**2xLxL
        seqmat_pad = np.pad(seqmat, ((0, 0), (1,rside_pad), (1, rside_pad)), constant_values=0)
        ret['input_seqmat'] = torch.FloatTensor(seqmat_pad) # NUM_BASE**2 x (Lmax+2) x (Lmax+2)
        seq_onehot_pad = np.pad(seq_onehot, ((1, rside_pad), (0, 0)), constant_values=0)
        ret['seq_onehot'] = torch.FloatTensor(seq_onehot_pad) # LxNUM_BASE

        # nc_map: noncanonical : (Lmax+2) x (Lmax+2)
        nc_map = seqmat[self.noncanonical_flag].sum(axis=0).astype(bool) # LxL
        nc_map_pad = np.pad(nc_map, ((1, rside_pad), (1, rside_pad)), constant_values=0)
        ret['nc_map'] = torch.FloatTensor(nc_map_pad)

        # BPPM: 1x(Lmax+2)x(Lmax+2)
        if self.use_BPP:
            bppm = self.load_BPPM(seq=seq, name=name, use_cache=(self.phase!='predict'))
            bppm_pad = np.pad(bppm, ((1, rside_pad), (1, rside_pad)), constant_values=0)
            ret['BPPM'] = torch.FloatTensor(bppm_pad).unsqueeze(0)
            # ret['BPPM'] = torch.log(ret['BPPM']+1e-5) # Note: Energy ~ log(p)
            # ret['BPPM'] = - ret['BPPM']
        # BPEM: 1x(Lmax+2)x(Lmax+2)
        if self.use_BPE:
            bpem = self.BPM_ene.get_energy(ret['seq'], normalize_energy=self.normalize_energy, BPM_type=self.BPM_type)
            bpem_pad = np.pad(bpem, ((0, 0), (1, rside_pad), (1, rside_pad)), constant_values=0)
            ret['BPEM'] = torch.FloatTensor(bpem_pad)

        y = {k: ret[k] for k in ['mask', 'forward_mask', 'nc_map', 'seq_onehot']}
        # gt, contact map: (Lmax+2)x(Lmax+2)
        if self.phase != 'predict':
            gt = connects2mat(connects)
            gt_pad = np.pad(gt, ((1, rside_pad), (1, rside_pad)), constant_values=0)
            ret['gt'] = torch.FloatTensor(gt_pad)
            y['gt'] = ret['gt']
        return ret, y

    def __getitem__(self, idx):
        info_dic = self.file_list[idx]
        dataset = info_dic['dataset'] if 'dataset' in info_dic else 'RNAseq'
        seq = name = connects = None
        
        if 'path' in info_dic:
            path = info_dic['path']
            name, suf = get_file_name(path, return_suf=True)
            if suf.lower() in {'.bpseq', '.ct', '.dbn'}:
                seq, connects = read_SS(path)
            else:
                with open(path) as fp:
                    fp.readline()
                    seq = fp.readline().strip(' \n')
        else:
            if 'seq' in info_dic:
                seq = info_dic['seq']
            else:
                raise Exception(f'[Error] seq or path needed: {info_dic}')
            name = info_dic['name'] if 'name' in info_dic else str_localtime()
        if connects is None and self.phase != 'predict':
            raise Exception(f'[Error] Invalid input: {info_dic} at {self.phase} stage, gt SS needed.')
        # # load_data
        # ret_data = {}
        # cache_path = os.path.join(self.cache_dir, name+'.pth')
        # if os.path.exists(cache_path):
        #     ret_data = torch.load(cache_path)
        # else:
        #     ret_data, y = self.prepare_data(name=name, seq=seq, connects=connects)
        #     if not self.mask_only:
        #         torch.save(ret_data, cache_path)
        ret_data, y = self.prepare_data(name=name, seq=seq, connects=connects)

        if self.mask_only:
            return {k: ret_data[k] for k in ['mask', 'forward_mask', 'length']}
        # update ret dic
        ret_data.update({'name': name, 'idx': idx, 'dataset': dataset})
        return ret_data, y

    def seq_embed_matrix(self, seq, return_onehot=False):
        ''' 
            seq: str, len=L, 'AUGC...'
            ret: tensor, NUM_BASE**2 x L x L, 0, 1 val
        '''
        L = len(seq)
        # seq onehot  L x NUM_BASE
        seq_onehot = np.zeros((L, self.num_base), dtype=float)
        for i in range(L): # should be consistent to function `postprocess` in ..util.postprocess
            seq_onehot[i][self.base_index[seq[i]]] = 1

        # seq embeding: NUM_BASE*NUM_BASE x L x L
        seq_embed = np.zeros((self.num_base**2, L, L)) # AUGC
        for n, (i, j) in enumerate(product(range(self.num_base), range(self.num_base))):
            seq_embed[n] = np.matmul(seq_onehot[:, i].reshape(-1, 1), seq_onehot[:, j].reshape(1, -1))
        if return_onehot:
            return seq_embed, seq_onehot
        else:
            return seq_embed

    def seq_embed_sequence(self, seq):
        ''' 
            seq: str, len=L, 'AUGC...'
            ret: ndarray, Lmax+2, 0-6 val, repr 'start AUGC... end empty...'
        '''
        ret = [self.token_index['START']]
        ret.extend(self.token_index[s] for s in seq)
        ret.append(self.token_index['END'])
        
        # Lmax + 2, final length
        for i in range(self.Lmax - len(seq)): 
            ret.append(self.token_index['EMPTY'])
        ret = torch.Tensor(ret).int() # float? TODO
        return ret


    def load_BPPM(self, seq, name, use_cache=True):

        txt_path = os.path.join(self.cache_dir, name+'.txt')
        npy_path = os.path.join(self.cache_dir, name+'.npy')

        if use_cache and os.path.exists(npy_path):
            return np.load(npy_path, allow_pickle=True)
        else:
            if not os.path.exists(txt_path):
                try:
                    if self.phase == 'predict' and self.verbose:
                        print(f'[Info] Using "{self.method}" to generate BPPM, saving at "{txt_path}"')
                    gen_BPPM(txt_path, mut_seq(seq), name, self.method)
                except Exception as e:
                    if self.phase == 'predict' and self.verbose:
                        print(f'[Warning] {e}, using CDPfold instead')
                    gen_BPPM(txt_path, mut_seq(seq), name, 'CDPfold')
            BPPM = read_BPPM(txt_path, len(seq))
            np.save(npy_path, BPPM)
            return BPPM
