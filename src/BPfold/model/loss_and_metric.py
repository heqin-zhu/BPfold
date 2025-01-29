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


def _cal_metric_from_tp(length, pred_p, gt_p, tp, eps=1e-12):
    '''
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
    fp = pred_p - tp
    fn = gt_p - tp
    recall = (tp + eps)/(tp+fn+eps)
    precision = (tp + eps)/(tp+fp+eps)
    f1_score = (2*tp + eps)/(2*tp + fp + fn + eps)
    tn = length - tp - fp - fn
    mcc = (tp * tn - fp * fn + eps)/(((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**0.5+eps)
    inf = (precision*recall)**0.5
    return mcc, inf, f1_score, precision, recall


def cal_metric(pred, gt):
    '''
        pred, gt: torch.Tensor
        return: MCC, INF, F1, precision, recall
    '''
    pred_p = torch.sign(pred).sum()
    gt_p = gt.sum()
    tp = torch.sign(pred*gt).sum()
    return _cal_metric_from_tp(torch.flatten(pred).shape[0], pred_p, gt_p, tp)


def cal_metric_pairwise(pred_pairs:[int], gt_pairs:[int]):
    '''
        pred_pairs, gt_pairs: connections, 1-indexed
        return: MCC, INF, F1, precision, recall
    '''
    length = len(pred_pairs)
    if length!=len(gt_pairs):
        raise Exception(f'[Error]: lengthes dismatch: pred {length}!= gt {len(gt_pairs)}')
    pred_p = gt_p = tp = 0 # predpair, gtpair, paired
    for pred, gt in zip(pred_pairs, gt_pairs):
        if gt!=0:
            gt_p +=1
        if pred!=0:
            pred_p +=1
            if pred==gt:
                tp +=1
    return _cal_metric_from_tp(length, pred_p, gt_p, tp)


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
        MCC, INF, f1, p, r = cal_metric(cur_pred, cur_gt)
        metric_dic[dataset_name][seq_name] = \
                {
                 'MCC': MCC.detach().cpu().numpy().item(),
                 'INF': INF.detach().cpu().numpy().item(),
                 'F1': f1.detach().cpu().numpy().item(),
                 'P': p.detach().cpu().numpy().item(),
                 'R': r.detach().cpu().numpy().item(),
        }
    return metric_dic


class myMetric(Metric):
    def __init__(self, metric_name='F1', device=None): 
        '''
            metric_name: F1, MCC
        '''
        self.reset()
        self.metric_name = metric_name.upper()
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
        self.metrics += [d[self.metric_name] for dic in metric_dic.values() for d in dic.values()]

    @property
    def value(self):
        return np.mean(self.metrics)
    
    @property
    def name(self):
        return self.metric_name
