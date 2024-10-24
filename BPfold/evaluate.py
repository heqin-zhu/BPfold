import os
import yaml
import tqdm
import argparse

import torch
import numpy as np

from .util.yaml_config import toYaml
from .util.misc import timer, get_file_name
from .util.RNA_kit import read_SS, connects2arr, dbn2connects, dispart_nc_pairs, merge_connects
from .model.loss_and_metric import cal_metric, cal_metric_pairwise


@timer
def evaluate(dest_path:str, pred_dir:str, gt_dir:str, pairwise:bool=True, read_pred=None, testsets=None)->None:
    '''
    Cal metrics according to SS files saved in pred_dir and gt_dir, and save results in dest_path (.yaml).

    Parameters
    ----------
    dest_path : str
        Path of yaml file to save the metric results
    pred_dir : str
        Path of pred SS files. Directory structure:
        -- pred_dir
            -- Dataset1
                -- file1.bpseq/.ct/.dbn/.out
                -- file2.bpseq/.ct/.dbn/.out
            -- Dataset2
    gt_dir : str
        Path of gt bpseq files. Directory structure is the same as pred_dir.
    '''
    SS_sufs = ['.bpseq', '.ct', '.dbn', '.out']
    SS_sufs += [i.upper() for i in SS_sufs]
    metric_dic = {}
    missing_pred = {}
    for dataset in os.listdir(gt_dir):
        if testsets is not None and dataset not in testsets:
            continue
        gt_data_dir = os.path.join(gt_dir, dataset)
        pred_data_dir = os.path.join(pred_dir, dataset)
        metric_dic[dataset] = {}
        for f in os.listdir(gt_data_dir):
            name = get_file_name(f)
            gt_path = os.path.join(gt_data_dir, f)
            pred_paths = [os.path.join(pred_pre, name+suf) for suf in SS_sufs for pred_pre in {pred_data_dir, pred_dir}]
            for pred_path in pred_paths:
                if os.path.exists(pred_path) and os.path.getsize(pred_path):
                    seq, pred_connects, gt_connects = get_seq_and_pred_gt_connects(pred_path, gt_path, read_pred)
                    canonical_pred, nc_pred = dispart_nc_pairs(seq, pred_connects)
                    canonical_gt, nc_gt = dispart_nc_pairs(seq, gt_connects)
                    metric_dic[dataset][name] = get_metric_dic(canonical_pred, canonical_gt, pairwise)
                    if dataset.startswith('PDB_test'): # or any([i!=0 for i in nc_pred]):
                        # nc: non-canonical
                        nc_pred_path = os.path.join(os.path.dirname(pred_path), get_file_name(pred_path) + '_nc.bpseq')
                        if os.path.exists(nc_pred_path): 
                            _, nc_pred = read_SS(nc_pred_path)
                            pred_connects = merge_connects(canonical_pred, nc_pred)
                        pred_gt_dic = {'': {'pred': canonical_pred, 'gt': canonical_gt}, 
                                       '_nc': {'pred': nc_pred, 'gt': nc_gt},
                                       '_mix': {'pred': pred_connects, 'gt': gt_connects},
                                      }
                        for flag in ['', '_nc', '_mix']:
                            cur_dataset_str = dataset + flag
                            if cur_dataset_str not in metric_dic:
                                metric_dic[cur_dataset_str] = {}
                            metric_dic[cur_dataset_str][name] = get_metric_dic(pred_gt_dic[flag]['pred'], pred_gt_dic[flag]['gt'], pairwise)
                    break
            else: # No pred results
                metric_dic[dataset][name] = {k: 0 for k in ['f1', 'p', 'r', 'length']}
                missing_pred[name] = {'name': name, 'pred_dir': pred_dir, 'dataset': dataset}
    if missing_pred:
        print('missing pred', len(missing_pred))
        miss_path = os.path.join(os.path.dirname(dest_path), get_file_name(dest_path)+'_missing.yaml')
        toYaml(miss_path, missing_pred)
    toYaml(dest_path, metric_dic)


def get_metric_dic(pred_connects, gt_connects, pairwise=True):
    length = len(gt_connects)
    if pairwise:
        f1, p, r = cal_metric_pairwise(pred_connects, gt_connects)
        return {'f1': f1, 'p': p, 'r': r, 'length': length}
    else:
        gt = connects2arr(gt_connects)
        pred = connects2arr(pred_connects)
        f1, p, r = cal_metric(torch.FloatTensor(pred), torch.FloatTensor(gt))
        return {
                'f1': f1.detach().cpu().numpy().item(), 
                'p': p.detach().cpu().numpy().item(), 
                'r': r.detach().cpu().numpy().item(), 
                'length': length,
                }


def get_seq_and_pred_gt_connects(pred_path, gt_path, read_pred=None):
    '''
        pred_path, gt_path: .bpseq, .ct, .dbn, .out
        return: pred_connects, gt_connects
    '''
    if read_pred is None:
        read_pred = read_SS
    gt_bases, gt_connects = read_SS(gt_path)
    gt_seq = ''.join(gt_bases).upper()
    length = len(gt_seq)
    pred_bases, pred_connects = read_pred(pred_path)
    pred_seq = ''.join(pred_bases).upper()
    if len(gt_seq) != len(pred_seq):
        raise Exception(f'[Error] Lengthes of seqs dismatch: \n{gt_path}: {gt_seq}\n{pred_path}: {pred_seq}')
    if gt_seq != pred_seq:
        print(f'[Warning] Seq bases dismatch: {gt_path}, {pred_path}')
    return pred_seq, pred_connects, gt_connects



def summary(path, to_latex=True):
    metric_dic_all = None
    with open(path) as fp:
        metric_dic_all = yaml.load(fp.read(), Loader=yaml.FullLoader)
    metric_dic_none_value = {}
    metric_dic = {}
    metric_dic_gt600 = {}
    metric_dic_le600 = {}
    for dataset, dic in metric_dic_all.items():
        metric_dic_none_value[dataset] = {}
        metric_dic[dataset] = {}
        metric_dic_le600[dataset] = {}
        metric_dic_gt600[dataset] = {}
        for name, d in dic.items():
            if d['f1'] is None:
                metric_dic_none_value[dataset][name] = d
            else:
                metric_dic[dataset][name] = d
                if d['length']<=600:
                    metric_dic_le600[dataset][name] = d
                else:
                    metric_dic_gt600[dataset][name] = d

    metric_names = ['f1', 'p', 'r']
    pred_and_all = sorted([(dataset, len(metric_dic[dataset]),len(d)) for dataset, d in metric_dic_all.items()])

    len_just = 12
    outputs = []
    outputs.append(f'[Summary] {path}')
    outputs.append(f' Pred/Total num: {pred_and_all}')
    for tag, cur_dic in [('all', metric_dic), ('gt600', metric_dic_gt600), ('le600', metric_dic_le600)]:
        outputs.append(f'-----{tag}-----')
        outputs.append(format_row(['dataset', 'num', 'f1', 'p', 'r'], len_just))
        for dataset_name, dic in cur_dic.items():
            if len(dic)!=0:
                vals = [(metric, np.mean([d[metric] for d in dic.values()])) for metric in metric_names]
                items = [dataset_name, str(len(dic)), *[f'{v[-1]:.3f}' for v in vals]]
                adjust_idx = {0, 1}
                outputs.append(format_row(items, len_just, adjust_idx))

    name = get_file_name(path)
    save_path = os.path.join(os.path.dirname(path), name+'_summary.txt')
    with open(save_path, 'w') as fp:
        for line in outputs:
            print(line)
            fp.write(line+'\n')


def format_row(items, len_just=12, adjust_idx=None):
    if adjust_idx is None:
        adjust_idx = {}
    return ' & '.join([str(item).ljust(len_just) if idx in adjust_idx else str(item) for idx, item in enumerate(items)])+'\\\\'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', type=str)
    parser.add_argument('--gt_dir', type=str, default='../mydata/test_data')
    parser.add_argument('--tag', type=str)
    parser.add_argument('--testsets', nargs='*')
    parser.add_argument('--summary', type=str, help='yaml file containing metric results.')
    args = parser.parse_args()
    if args.pred_dir:
        if args.tag is None:
            segs = args.pred_dir.split(os.path.sep)[-3:]
            args.tag = 'eval_'+ '_'.join(segs)

        dest_path = args.tag+'.yaml'
        evaluate(dest_path, args.pred_dir, args.gt_dir, testsets=args.testsets)
        summary(dest_path)
    elif args.summary:
        summary(args.summary)
