from .BPfoldNet import BPfold_model
from .loss_and_metric import BCE_loss, MSE_loss, myMetric, cal_metric, cal_metric_batch

def get_model(s):
    return {
            'bpfold': BPfold_model,
            'bpfold_model': BPfold_model,
           }[s.lower()]

def  get_loss(s):
    return {
            'bce': BCE_loss,
            'mse': MSE_loss,
           }[s.lower()]
