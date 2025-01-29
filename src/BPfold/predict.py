import os
import argparse
import random

import numpy as np

import torch
from torch.utils.data import DataLoader

from .core import load_eval_checkpoints, seed_everything
from .dataset import get_dataset
from .model import get_model
from .util.misc import get_file_name, str_localtime
from .util.hook_features import hook_features
from .util.yaml_config import get_config, read_yaml, write_yaml
from .util.postprocess import postprocess, apply_constraints
from .util.data_sampler import DeviceMultiDataLoader
from .util.RNA_kit import read_SS, write_SS, read_fasta, write_fasta, connects2dbn, arr2connects, compute_confidence, remove_lone_pairs, merge_connects


SRC_DIR = os.path.dirname(os.path.realpath(__file__))


class BPfold_predict:
    def __init__(self, checkpoint_dir):
        '''
        Init

        Parameters
        ----------
        checkpoint_dir: str
            Directory of checkpoints that contain trained parameters.
        '''
        self.tmp_dir = '.BPfold_tmp_files'
        os.makedirs(self.tmp_dir, exist_ok=True)

        self.para_dir = os.path.join(SRC_DIR, 'paras')
        config_file = os.path.join(SRC_DIR, 'configs/config.yaml')
        opts = get_config(config_file)
        model_name = opts['model']['model_name']
        data_name = opts['dataset']['data_name']
        data_opts = opts['dataset'][data_name]
        model_opts = opts['model'][model_name]
        common_opts = opts['common']
        data_opts.update(common_opts)
        model_opts.update(common_opts)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # load checkpoints
        RNA_model = get_model(model_name)
        self.models = load_eval_checkpoints(checkpoint_dir, RNA_model, model_opts, self.device)

        self.data_opts = data_opts

    def predict(self, input_seqs=None, input_path=None, batch_size=1, num_workers=1, hook_features=False, save_contact_map=False):
        '''
        BPfold `predict function`, specify input_seqs or input_path 

        Parameters
        ----------
        input_seqs: str | [str]
            RNA seqs
        input_path: str
            fasta path containing multi RNA seqs or Secondary structure path in format of 'dbn', 'bpseq' or 'ct'.

        Returns
        -------
        ret: [str, str, [int], float]
            seq_name, seq, connects, CI
        '''
        dl = self.get_predict_loader(self.data_opts, self.device, input_seqs, input_path, batch_size, num_workers, data_name='RNAseq')
        for data_dic, _ in dl:
            with torch.no_grad(),torch.cuda.amp.autocast():
                # torch.nan_to_num
                # BS x forward_batch_Lmax x forward_batch_Lmax
                pred_batch = torch.stack([model(data_dic) for model in self.models], 0).mean(0)

                # remove `begin` and `end` tokens
                forward_batch_Lmax = data_dic['forward_mask'].sum(-1).max()
                batch_Lmax = forward_batch_Lmax-2
                pred_batch = pred_batch[:, 1:batch_Lmax+1, 1:batch_Lmax+1]
                seq_onehot = data_dic['seq_onehot'][:, 1:batch_Lmax+1, :]
                nc_map = data_dic['nc_map'][:, 1:batch_Lmax+1, 1:batch_Lmax+1]
                masks = data_dic['mask'][:, 1:batch_Lmax+1, 1:batch_Lmax+1]
                seqs = data_dic['ori_seq']
                names = data_dic['name']

                if hook_features:
                    # hook_dir
                    hook_dir = os.path.join(self.tmp_dir, 'hook_features')
                    os.makedirs(hook_dir, exist_ok=True)
                    hook_module_names = ['TransformerEncoderLayer', 'ResConv2dSimple']
                    hooker = hook_features(self.models[0], hook_module_names)
                    module_count = {}
                    for module_name, input_feature, output_feature in zip(*hooker.get_hook_results()):
                        if module_name not in module_count:
                            module_count[module_name] = 0
                        module_count[module_name]+=1
                        save_name = f'{names[0]}_{module_name}_{module_count[module_name]:02d}'
                        save_path = os.path.join(hook_dir, save_name + '.npy')
                        out_map = output_feature[0].detach().cpu().numpy()
                        np.save(save_path, out_map)

                # postprocess
                ret_pred, ret_pred_nc, _, _ = postprocess(pred_batch, seq_onehot, nc_map, return_score=False, return_nc=True)
                # save pred
                for i in range(len(ret_pred)):
                    length = len(seqs[i])
                    seq_name = names[i]
                    mat = pred_batch[i][masks[i]].reshape(length, length).detach().cpu().numpy()
                    mat_post = ret_pred[i][masks[i]].reshape(length, length).detach().cpu().numpy()
                    CI = compute_confidence(mat, mat_post)
                    connects = arr2connects(mat_post)
                    connects = remove_lone_pairs(connects)

                    ## save contact maps before and after postprocessing
                    if save_contact_map:
                        save_data_dir = os.path.join(self.tmp_dir, 'contact_map')
                        os.makedirs(save_data_dir, exist_ok=True)
                        ## save numpy arr, before/after postprocessing
                        mat = pred_batch[i][masks[i]].reshape(length, length).detach().cpu().numpy()
                        mat_post = ret_pred[i][masks[i]].reshape(length, length).detach().cpu().numpy()
                        np.save(os.path.join(save_data_dir, f'{seq_name}.npy'), mat)
                        np.save(os.path.join(save_data_dir, f'{seq_name}_post.npy'), mat_post)
                        post_before_th = apply_constraints(pred_batch[i:i+1], seq_onehot[i:i+1], 0.01, 0.1, 100, 1.6, True, 1.5)[0]
                        np.save(os.path.join(save_data_dir, f'{seq_name}_post_before_th.npy'), post_before_th[masks[i]].reshape(length, length).detach().cpu().numpy())

                    yield {'seq_name': seq_name, 'seq': seqs[i], 'connects': connects, 'CI': CI}


    def gen_info_dic(self, input_seqs, input_path, data_name='RNAseq'):
        def valid_seq(seq):
            return seq.isalpha()
        def process_one_seq(name, seq, file_path=''):
            tmp_path = os.path.join(tmp_seq_dir, name+'.fasta')
            write_fasta(tmp_path, [(name, seq)])
            if valid_seq(seq):
                return {'path': tmp_path, 'name': name, 'length': len(seq), 'dataset': data_name}
            else:
                print(f'[Warning] Invalid seq, containing non-alphabet character, ignored: {file_path}-{name}="{seq}"')
        def process_one_file(file_path, data_name='RNAseq'):
            file_name, suf = get_file_name(file_path, return_suf=True)
            if suf.lower() in {'.fasta', '.fa'}: # fasta file
                for name, seq in read_fasta(file_path):
                    info_dic = process_one_seq(name, seq, file_path)
                    if info_dic is not None:
                        yield info_dic
            elif suf.lower() in {'.dbn', '.ct', '.bpseq'}: # SS file
                seq, _ = read_SS(file_path)
                yield {'path': file_path, 'length': len(seq), 'dataset': data_name}
            else: # unknown filetype
                print(f'[Warning] Unknown file type, ignored: {file_path}')

        tmp_seq_dir = os.path.join(self.tmp_dir, 'seq_fasta')
        os.makedirs(tmp_seq_dir, exist_ok=True)
        if input_seqs:
            time_str = str_localtime()
            for idx, seq in enumerate(input_seqs):
                seq_name = f'seq_{time_str}_{idx+1}'
                info_dic = process_one_seq(seq_name, seq)
                if info_dic:
                    yield info_dic
        if input_path:
            if os.path.isfile(input_path):
                yield from process_one_file(input_path, data_name)
            else:
                for pre, ds, fs in os.walk(input_path):
                    for f in fs:
                        yield from process_one_file(os.path.join(pre, f), data_name)

    def get_predict_loader(self, data_opts, device, input_seqs, input_path, batch_size, num_workers, data_name='RNAseq'):
        data_class = get_dataset(data_name)
        data_opts['predict_files'] = list(self.gen_info_dic(input_seqs, input_path, data_name) )
        ds = data_class(phase='predict', verbose=False, para_dir=self.para_dir, **data_opts)
        dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
        return DeviceMultiDataLoader([dl], device, keywords=ds.to_device_keywords)


def show_examples():
    method_name = 'BPfold'
    print('Please specify "--seq" or "--input" argument for input sequences or input file. Such as:')
    print(f'$ {method_name} --checkpoint_dir PATH_TO_CHECKPOINT --seq GGUAAAACAGCCUGU AGUAGGAUGUAUAUG --output {method_name}_results')
    print(f'$ {method_name} --checkpoint_dir PATH_TO_CHECKPOINT --input examples/examples.fasta # (multiple sequences are supported)')
    print(f'$ {method_name} --checkpoint_dir PATH_TO_CHECKPOINT --input examples/URS0000D6831E_12908_1-117.bpseq # .bpseq, .ct, .dbn')
    exit()


def save_pred(seq, connects, save_types, name, pred_dir):
    ret = []
    for out_type in save_types:
        path = os.path.join(pred_dir, name+f'.{out_type}')
        write_SS(path, seq, connects, out_type=out_type)
        ret.append(path)
    return ret


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint_dir', type=str, help='Directory of checkpoints that contain trained parameters.', required=True)
    parser.add_argument('-s', '--seq', nargs='*', help='RNA sequences')
    parser.add_argument('-i', '--input', type=str, help='Input fasta file or directory which contains fasta files, supporting multiple seqs and multiple formats, such as fasta, bpseq, ct or dbn.')
    parser.add_argument('-o', '--output', type=str, default='BPfold_results', help='output directory')
    parser.add_argument('-g', '--gpu', type=str, default='0')
    parser.add_argument('--save_type', default='bpseq', choices=['bpseq', 'ct', 'dbn', 'all'], help='Saved file type.')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--hide_dbn', action='store_true', help='Once specified, the output sequence and predicted DBN won\'t be printed.')
    parser.add_argument('--save_nc', action='store_true', help='Additionally save prediction with non-canonical pairs.')
    parser.add_argument('--save_contact_map', action='store_true')
    parser.add_argument('--hook_features', action='store_true')
    args = parser.parse_args()
    return args


def main():
    print('>> Welcome to use "BPfold" for predicting RNA secondary structure!')
    args = parse_args() 
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    seed_everything(42)

    pred_dir = args.output
    os.makedirs(pred_dir, exist_ok=True)
    confidence_path = os.path.join(os.path.dirname(pred_dir), os.path.basename(pred_dir)+'_confidence.yaml')
    confidence_dic = {}
    if os.path.exists(confidence_path):
        confidence_dic = read_yaml(confidence_path)

    num_digit = 5
    save_types = [args.save_type] if args.save_type!='all' else ['bpseq', 'ct', 'dbn']

    # usage
    if args.input is None and args.seq is None:
        show_examples()

    BPfold_predictor = BPfold_predict(checkpoint_dir=args.checkpoint_dir)
    pred_results = BPfold_predictor.predict(args.seq, args.input, args.batch_size, args.num_workers, args.hook_features, args.save_contact_map)
    for ct, res_dic in enumerate(pred_results):
        seq_name = res_dic['seq_name']
        seq = res_dic['seq']
        connects = res_dic['connects']
        CI = res_dic['CI']
        confidence_dic[seq_name] = CI
        path = save_pred(seq, connects, save_types, seq_name, pred_dir)[0]
        seq, connects = read_SS(path)
        CI_str = f'CI={CI:.3f}' if CI>=0.3 else 'CI<0.3'
        print(f"[{str(ct+1).rjust(num_digit)}] saved in \"{path}\", {CI_str}")
        if not args.hide_dbn:
            print(f'{seq}\n{connects2dbn(connects)}')
        if args.save_nc:
            mat_nc_post = ret_pred_nc[i][masks[i]].reshape(length, length).detach().cpu().numpy()
            connects_nc = arr2connects(mat_nc_post)
            save_pred(seq, connects_nc, save_types, seq_name+'_nc', pred_dir)[0]
            connects_mix = merge_connects(connects, connects_nc)
            save_pred(seq, connects_mix, save_types, seq_name+'_mix', pred_dir)[0]
            if not args.hide_dbn:
                print(f'{connects2dbn(connects_nc)} NC')
                print(f'{connects2dbn(connects_mix)} MIX')
    print(f"Confidence index saved in \"{confidence_path}\"")
    write_yaml(confidence_path, confidence_dic)
    print('Finished!')


if __name__ == '__main__':
    main()
