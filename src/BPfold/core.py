import os, gc
import random
import argparse

from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
# Fix fastai bug to enable fp16 training with dictionaries

import fastai
from fastai.vision.all import Callback, L, to_float, CancelStepException, delegates, DataLoaders
from fastai.vision.all import SaveModelCallback, EarlyStoppingCallback, GradientClip, Learner

from .dataset import get_dataset
from .model import get_model, get_loss, myMetric, cal_metric_batch
from .util.yaml_config import write_yaml, get_config, update_config
from .util.postprocess import postprocess, apply_constraints
from .util.data_sampler import LenMatchBatchSampler, DeviceMultiDataLoader
from .util.RNA_kit import write_SS, arr2connects, remove_lone_pairs


def flatten(o):
    "Concatenate all collections and items as a generator"
    for item in o:
        if isinstance(o, dict): yield o[item]; continue
        elif isinstance(item, str): yield item; continue
        try: yield from flatten(item)
        except TypeError: yield item


@delegates(GradScaler)
class MixedPrecision(Callback):
    "Mixed precision training using Pytorch's `autocast` and `GradScaler`"
    order = 10
    def __init__(self, **kwargs): self.kwargs = kwargs
    def before_fit(self): 
        self.autocast,self.learn.scaler,self.scales = autocast(),GradScaler(**self.kwargs),L()
    def before_batch(self): self.autocast.__enter__()
    def after_pred(self):
        if next(flatten(self.pred)).dtype==torch.float16: self.learn.pred = to_float(self.pred)
    def after_loss(self): self.autocast.__exit__(None, None, None)                       
    def before_backward(self): self.learn.loss_grad = self.scaler.scale(self.loss_grad)
    def before_step(self):
        "Use `self` as a fake optimizer. `self.skipped` will be set to True `after_step` if gradients overflow. "
        self.skipped=True
        self.scaler.step(self)
        if self.skipped: raise CancelStepException()
        self.scales.append(self.scaler.get_scale())
    def after_step(self): self.learn.scaler.update()

    @property 
    def param_groups(self): 
        "Pretend to be an optimizer for `GradScaler`"
        return self.opt.param_groups
    def step(self, *args, **kwargs): 
        "Fake optimizer step to detect whether this batch was skipped from `GradScaler`"
        self.skipped=False
    def after_fit(self): self.autocast,self.learn.scaler,self.scales = None,None,None
fastai.callback.fp16.MixedPrecision = MixedPrecision
        

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def load_eval_checkpoints(ckpt_dir, RNA_model, model_opts, device, ckpt_names=None):
    if not os.path.exists(ckpt_dir):
        raise Exception(f'[Error] Checkpoint directory not exist: {ckpt_dir}')
    models = []
    if ckpt_names is None:
        ckpt_names = sorted(os.listdir(ckpt_dir))
    for ckpt_name in ckpt_names:
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        print(f'Loading {os.path.abspath(ckpt_path)}')
        model = RNA_model(**model_opts)
        model = model.to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu')))
        model.eval()
        models.append(model)
    if models == []:
        raise Exception(f'[Error] No checkpoint found in {ckpt_dir}')
    return models
