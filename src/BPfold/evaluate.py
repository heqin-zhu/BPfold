import os
import yaml
import tqdm
import argparse

import torch
import numpy as np

from .util.yaml_config import read_yaml, write_yaml
from .util.misc import timer, get_file_name
from .util.RNA_kit import read_SS, dispart_nc_pairs, merge_connects, cal_metric


def get_index_data(gt_dir, testsets):
    index_path = None
    for index_name in ['data_index.yaml']:
        cur_path = os.path.join(gt_dir, index_name)
        if os.path.exists(cur_path):
            index_path = cur_path
            break
    index_data = {}
    if index_path is None:
        for dataset in os.listdir(gt_dir):
            dataset_dir = os.path.join(gt_dir, dataset)
            if os.path.isdir(dataset_dir):
                if dataset.startswith('PDB_test'):
                    dataset = 'PDB_test'
                if testsets and dataset not in testsets:
                    continue
                if dataset not in index_data:
                    index_data[dataset] = []

                for pre, ds, fs in os.listdir(dataset_dir):
                    for f in fs:
                        if f.endswith('.bpseq'):
                            index_data[dataset].append(os.path.join(pre, f))
    else:
        for dic in read_yaml(index_path)['test']:
            dataset = dic['dataset']
            if dataset.startswith('PDB_test'):
                dataset = 'PDB_test'
            if testsets is not None and dataset not in testsets:
                continue
            if dataset not in index_data:
                index_data[dataset] = []
            index_data[dataset].append(os.path.join(gt_dir, dic['path']))
    return index_data


@timer
def evaluate(dest_path:str, pred_dir:str, gt_dir:str, read_pred=None, testsets=None, show_nc=False)->None:
    '''
    Cal metrics of predicted SS in pred_dir and gt_dir, then save results in dest_path (.yaml).

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
    for dataset, gt_files in get_index_data(gt_dir, testsets).items():
        pred_data_dir = os.path.join(pred_dir, dataset)
        metric_dic[dataset] = {}
        for gt_path in gt_files:
            name = get_file_name(gt_path)
            pred_paths = [os.path.join(pred_pre, name+suf) for suf in SS_sufs for pred_pre in {pred_data_dir, pred_dir}]
            for pred_path in pred_paths:
                if os.path.exists(pred_path) and os.path.getsize(pred_path):
                    seq, pred_connects, gt_connects = get_seq_and_pred_gt_connects(pred_path, gt_path, read_pred)
                    canonical_pred, nc_pred = dispart_nc_pairs(seq, pred_connects)
                    canonical_gt, nc_gt = dispart_nc_pairs(seq, gt_connects)
                    metric_dic[dataset][name] = get_metric_dic(canonical_pred, canonical_gt)
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
                        types = ['', '_nc', '_mix'] if show_nc else ['']
                        for flag in types:
                            cur_dataset_str = dataset + flag
                            if cur_dataset_str not in metric_dic:
                                metric_dic[cur_dataset_str] = {}
                            metric_dic[cur_dataset_str][name] = get_metric_dic(pred_gt_dic[flag]['pred'], pred_gt_dic[flag]['gt'])
                    break
            else: # No pred results
                metric_dic[dataset][name] = {k: None for k in ['MCC', 'INF', 'F1', 'Precision', 'Recall', 'length']}
                missing_pred[name] = {'name': name, 'pred_dir': pred_dir, 'dataset': dataset}
    if missing_pred:
        print('missing pred', len(missing_pred))
        miss_path = os.path.join(os.path.dirname(dest_path), get_file_name(dest_path)+'_missing.yaml')
        write_yaml(miss_path, missing_pred)
    write_yaml(dest_path, metric_dic)


def get_metric_dic(pred_connects, gt_connects):
    length = len(gt_connects)
    m_dic = cal_metric(pred_connects, gt_connects)
    m_dic['length'] = length
    return m_dic

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
    metric_dic = {}
    metric_dic_gt600 = {}
    metric_dic_le600 = {}
    for dataset, dic in metric_dic_all.items():
        metric_dic[dataset] = {}
        metric_dic_le600[dataset] = {}
        metric_dic_gt600[dataset] = {}
        for name, d in dic.items():
            d = {k: 0 if v is None else v for k,v in d.items()}
            metric_dic[dataset][name] = d
            if d['length']<=600:
                metric_dic_le600[dataset][name] = d
            else:
                metric_dic_gt600[dataset][name] = d

    metric_names = ['INF', 'F1', 'Precision', 'Recall']
    pred_and_all = sorted([(dataset, len(metric_dic[dataset]),len(d)) for dataset, d in metric_dic_all.items()])

    outputs = []
    outputs.append(f'[Summary] {path}')
    outputs.append(f' Pred/Total num: {pred_and_all}')
    for tag, cur_dic in [('len>600', metric_dic_gt600), ('len<=600', metric_dic_le600), ('all', metric_dic)]:
        title_row = format_row(['dataset', 'num', 'INF', 'F1', 'Precision', 'Recall'], [15, 5, 5, 5, 5, 5])
        length = (len(title_row) - len(tag))//2
        outputs.append('-'*length + tag + '-'*(len(title_row) - length - len(tag)))
        outputs.append(title_row)
        for dataset_name, dic in cur_dic.items():
            if len(dic)!=0:
                vals = [(metric, np.mean([d[metric] for d in dic.values()])) for metric in metric_names]
                items = [dataset_name, str(len(dic)), *[f'{v[-1]:.3f}' for v in vals]]
                outputs.append(format_row(items, [15, 5, 5, 5, 5, 5]))

    name = get_file_name(path)
    save_path = os.path.join(os.path.dirname(path), name+'_summary.txt')
    with open(save_path, 'w') as fp:
        for line in outputs:
            print(line)
            fp.write(line+'\n')


def format_row(items, len_adjust=None):
    if len_adjust is None:
        len_adjust = [0]*len(items)
    return ' & '.join([str(item).ljust(adj) for item, adj in zip(items, len_adjust)])+'\\\\'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', type=str)
    parser.add_argument('--gt_dir', type=str, default='BPfold_data/test_data')
    parser.add_argument('--tag', type=str)
    parser.add_argument('--testsets', nargs='*')
    parser.add_argument('--show_nc', action='store_true')
    parser.add_argument('--summary', type=str, help='yaml file containing metric results.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.pred_dir:
        if args.tag is None:
            segs = args.pred_dir.split(os.path.sep)[-3:]
            args.tag = 'eval_'+ '_'.join(segs)

        dest_path = args.tag+'.yaml'
        evaluate(dest_path, args.pred_dir, args.gt_dir, testsets=args.testsets, show_nc=args.show_nc)
        summary(dest_path)
    elif args.summary:
        summary(args.summary)


if __name__ == '__main__':
    main()
