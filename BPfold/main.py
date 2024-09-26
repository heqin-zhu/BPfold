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
from .model import get_model, get_loss, F1_Metric, cal_metric_batch
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
        print(f'Loading {ckpt_path}')
        model = RNA_model(**model_opts)
        model = model.to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu')))
        model.eval()
        models.append(model)
    if models == []:
        raise Exception(f'[Error] No checkpoint found in {ckpt_dir}')
    return models


def train(opts):
    raise Exception('Training code will be released soon.')


def test(opts):
    raise Exception('Test code will be released soon.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--phase', choices=['train', 'test'], default='train')
    # basic configuration
    parser.add_argument('-c', '--config', type=str, default='configs/config.yaml')
    parser.add_argument('-g', '--gpu', type=str, default='0')
    parser.add_argument('--ignore_fold', action='store_true')
    parser.add_argument('--run_name', type=str, default='BPfold_dim256')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test_epoch', nargs='*', help='epochs of checkpoints corr to fold at test stage')

    # dataset configuration
    parser.add_argument('--data_name', type=str)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--Lmax', type=int, help='max length of RNA seq, [Lmin, Lmax]')
    parser.add_argument('--Lmin', type=int, help='min length of RNA seq, [Lmin, Lmax]')
    parser.add_argument('--cache_dir', type=str)
    parser.add_argument('--index_name', type=str)
    parser.add_argument('--method', choices=['EternaFold', 'CDPfold', 'Contrafold', 'ViennaRNA'])
    parser.add_argument('--trainall', action='store_true')
    parser.add_argument('--normalize_energy', action='store_true')
    parser.add_argument('--training_set', nargs='*', default=None)
    parser.add_argument('--test_set', nargs='*', default=None)

    # learning paras, training
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--earlystop', action='store_true')
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--fold_list', nargs='*', help='fold num list, 0 <= num < nfolds')
    parser.add_argument('--gradientclip', type=float)
    parser.add_argument('--load_checkpoint', action='store_true', help='load checkpoint when training')
    parser.add_argument('--loss', choices=['BCE', 'MSE'])
    parser.add_argument('--lr', type=float)

    parser.add_argument('--nfolds', type=int, help='nfolds<=1 for no kfold')
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--pos_weight', type=float)
    parser.add_argument('--save_freq', type=int)

    # model paras
    parser.add_argument('--model_name', choices=['BPfold'])
    parser.add_argument('--depth', type=int)
    parser.add_argument('--dim', type=int)
    parser.add_argument('--head_size', type=int)
    parser.add_argument('--not_slice', action='store_true')
    parser.add_argument('--positional_embedding', choices=['dyn', 'alibi'])
    
    parser.add_argument('--use_BPE', action='store_true')
    parser.add_argument('--use_BPP', action='store_true')
    # End

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    opts = get_config(args.config)
    update_config(opts, args)

    seed_everything(opts.common.seed)

    phase = args.phase
    run_name = opts.basic.run_name
    os.makedirs(run_name, exist_ok=True)
    write_yaml(os.path.join(run_name, f"config_{phase}.yaml"), opts)
    print(opts)
    if phase == 'train':
        train(opts)
    test(opts)
