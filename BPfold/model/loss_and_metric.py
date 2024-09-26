from functools import partial

import torch
import numpy as np
from fastai.vision.all import Metric

from ..util.postprocess import postprocess


def cal_loss(loss_func, pred, gt_dic, **args):
    '''
        pred: batch x batch_Lmax x batch_Lmax
        gt : batch x Lmax x Lmax
    '''
    forward_batch_Lmax = gt_dic['forward_mask'].sum(-1).max()
    batch_Lmax = forward_batch_Lmax - 2
    loss = loss_func(pred[:, 1:batch_Lmax+1, 1:batch_Lmax+1], gt_dic['gt'][:, 1:batch_Lmax+1, 1:batch_Lmax+1])
    return loss


def MSE_loss(pos_weight=300, **args):
    return partial(cal_loss, torch.nn.MSELoss(**args))


def BCE_loss(pos_weight=300, device=None, **args):
    pos_weight = torch.Tensor([pos_weight])
    if device is not None:
        pos_weight = pos_weight.to(device)
    loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, **args)
    return partial(cal_loss, loss_func)


def cal_metric(pred, gt, eps=1e-12):
    '''
        pred, gt: torch.Tensor
        return: F1, precision, recall

        reference:
            accuracy: acc = (TP+TN)/(TP+FP+FN+TN)
            precision: p = TP/(TP+FP)
            recall: r = TP/(TP+FN)
            F1: F1 = 2*p*r / (p+r)
            sensitivity = recal = TPR (true positive rate)
            specificity = TN/(TN+FP)
            YoudenIndex = sen + spe - 1
            false positive rate: FPR = FP/(TN+FP) = 1-spe
            positive predicted value: PPV = precision
            negative predicted value: NPV = TN/(TN+FN)
    '''
    tp_map = torch.sign(pred*gt)
    tp = tp_map.sum()
    pred_p = torch.sign(pred).sum()
    gt_p = gt.sum()
    fp = pred_p - tp
    fn = gt_p - tp
    recall = (tp + eps)/(tp+fn+eps)
    precision = (tp + eps)/(tp+fp+eps)
    f1_score = (2*tp + eps)/(2*tp + fp + fn + eps)
    return f1_score, precision, recall


def cal_metric_pairwise(pred_pairs:[int], gt_pairs:[int]):
    '''
        pred_pairs, gt_pairs: connections, 1-indexed
        return: F1, precision, recall

        reference:
            accuracy: acc = (TP+TN)/(TP+FP+FN+TN)
            precision: p = TP/(TP+FP)
            recall: r = TP/(TP+FN)
            F1: F1 = 2*p*r / (p+r)
            sensitivity = recall = TPR (true positive rate)
            specificity = TN/(TN+FP)
            YoudenIndex = sen + spe - 1
            false positive rate: FPR = FP/(TN+FP) = 1-spe
            positive predicted value: PPV = precision
            negative predicted value: NPV = TN/(TN+FN)
    '''
    n_pred = len(pred_pairs)
    n_gt = len(gt_pairs)
    if n_pred != n_gt:
        raise Exception(f'[Error]: lengthes dismatch: pred {n_pred}!= gt {n_gt}')
    n_paired = n_gtpair = n_predpair = 0
    for pred, gt in zip(pred_pairs, gt_pairs):
        if gt!=0:
            n_gtpair +=1
        if pred!=0:
            n_predpair +=1
            if pred==gt:
                n_paired +=1
    p = 1 if n_predpair==0 else n_paired/n_predpair
    r = 1 if n_gtpair==0 else n_paired/n_gtpair
    f1 = 0 if n_paired==0 else 2*p*r/(p+r)
    return f1, p, r



def cal_metric_batch(pred, gt, mask=None, seq_names=None, dataset_names=None):
    n = len(pred)
    if dataset_names is None:
        dataset_names = ['dataset' for i in range(n)]
    if seq_names is None:
        seq_names = [f'seq{i}' for i in range(n)]
    metric_dic = {dataset_name: {} for dataset_name in dataset_names}
    for i in range(n):
        dataset_name = dataset_names[i]
        seq_name = seq_names[i]
        cur_pred = pred[i] if mask is None else pred[i][mask[i]]
        cur_gt = gt[i] if mask is None else gt[i][mask[i]]
        f1, p, r = cal_metric(cur_pred, cur_gt)
        metric_dic[dataset_name][seq_name] = \
                {
                 'f1': f1.detach().cpu().numpy().item(),
                 'p': p.detach().cpu().numpy().item(),
                 'r': r.detach().cpu().numpy().item(),
        }
    return metric_dic


class F1_Metric(Metric):
    def __init__(self, device=None): 
        self.reset()
        self.cal_func = cal_metric_batch
        
    def reset(self): 
        self.metrics = []
        
    def accumulate(self, learn):
        pred_batch = learn.pred
        data_dic = learn.y
        
        # prepare
        forward_batch_Lmax = data_dic['forward_mask'].sum(-1).max()
        batch_Lmax = forward_batch_Lmax-2
        pred_batch = pred_batch[:, 1:batch_Lmax+1, 1:batch_Lmax+1]
        mask = data_dic['mask'][:, 1:batch_Lmax+1, 1:batch_Lmax+1]
        seq_onehot = data_dic['seq_onehot'][:, 1:batch_Lmax+1, :]
        gt = data_dic['gt'][:, 1:batch_Lmax+1, 1:batch_Lmax+1]
        nc_map = data_dic['nc_map'][:, 1:batch_Lmax+1, 1:batch_Lmax+1]
        # postprocess
        ret_pred, _, ret_score, _ = postprocess(pred_batch, seq_onehot, nc_map, return_nc=False, return_score=False)
        metric_dic = self.cal_func(ret_pred, gt, mask)
        self.metrics += [d['f1'] for dic in metric_dic.values() for d in dic.values()]

    @property
    def value(self):
        return np.mean(self.metrics)
    
    @property
    def name(self):
        return 'F1_Metric'
