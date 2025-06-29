from functools import partial

import torch
import numpy as np
from fastai.vision.all import Metric

from ..util.postprocess import postprocess
from ..util.RNA_kit import _cal_metric_from_tp


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


def cal_metric_torch(pred, gt):
    '''
        pred, gt: torch.Tensor
        return: MCC, INF, F1, precision, recall
    '''
    pred_p = torch.sign(pred).sum()
    gt_p = gt.sum()
    tp = torch.sign(pred*gt).sum()
    return _cal_metric_from_tp(torch.flatten(pred).shape[0], pred_p, gt_p, tp)


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
        m_dic = cal_metric_torch(cur_pred, cur_gt)
        metric_dic[dataset_name][seq_name] = {k: v.detach().cpu().numpy().item() for k, v in m_dic.items()}
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


def  get_loss(s):
    return {
            'bce': BCE_loss,
            'mse': MSE_loss,
           }[s.lower()]
