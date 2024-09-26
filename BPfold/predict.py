import os
import argparse
import random

import numpy as np

import torch
from torch.utils.data import DataLoader

from .main import load_eval_checkpoints, seed_everything
from .dataset import get_dataset
from .model import get_model
from .util.misc import get_file_name, str_localtime
from .util.hook_features import hook_features
from .util.yaml_config import read_yaml, get_config
from .util.postprocess import postprocess
from .util.data_sampler import DeviceMultiDataLoader
from .util.RNA_kit import read_SS, write_SS, read_fasta, connects2dbn, arr2connects, compute_confidence, remove_lone_pairs


def predict(args):
    def show_help_info(input_seqs, input_path, module_name='BPfold'):
        print(f'>> Welcome to use "{module_name}" for predicting RNA secondary structure!')
        if input_seqs is None and input_path is None:
            print('Please specify "--seq" or "--input" argument for input sequence. Such as:')
            print(f'$ python3 -m {module_name}.predict --seq GGUAAAACAGCCUGU --output {module_name}_results # output directory')
            print(f'$ python3 -m {module_name}.predict --input examples/1A9L.fasta # (multiple sequences are supported)')
            print(f'$ python3 -m {module_name}.predict --input examples/1A9L.bpseq # .bpseq, .ct, .dbn, or any other text file (Only the first line will be read as input sequence)')
            exit()

    def valid_seq(seq):
        return seq.isalpha()

    def gen_info_dic(input_seqs, input_path, data_name='RNAseq'):
        def process_one_file(file_path, data_name='RNAseq'):
            file_name, suf = get_file_name(file_path, return_suf=True)
            if suf.lower() in {'.fasta', '.fa'}: # fasta file
                for name, seq in read_fasta(file_path):
                    if valid_seq(seq):
                        yield name, {'seq': seq, 'name': name, 'length': len(seq), 'dataset': data_name}
                    else:
                        print(f'[Warning] Invalid seq, containing non-alphabet character, ignored: {file_path}-{name}="{seq}"')
            else:
                seq = None
                if suf.lower() in {'.dbn', '.ct', '.bpseq'}: # SS file
                    seq, _ = read_SS(file_path)
                else: # other text file
                    with open(file_path) as fp:
                        seq = fp.readline().strip(' \n')
                if valid_seq(seq):
                    yield file_name, {'seq': seq, 'name': file_name, 'length': len(seq), 'dataset': data_name}
                else:
                    print(f'[Warning] Invalid seq, containing non-alphabet character, ignored: {file_path}="{seq}"')
        if input_seqs:
            time_str = str_localtime()
            for idx, seq in enumerate(input_seqs):
                seq_name = f'seq_{time_str}_{idx+1}'
                if valid_seq(seq):
                    yield seq_name, {'seq': seq, 'name': seq_name, 'length': len(seq), 'dataset': data_name}
                else:
                    print(f'[Warning] Invalid seq, non-alphabet character, ignored: "{seq}"')
        if input_path:
            if os.path.isfile(input_path):
                yield from process_one_file(input_path, data_name)
            else:
                for pre, ds, fs in os.walk(input_path):
                    for f in fs:
                        yield from process_one_file(os.path.join(pre, f), data_name)


    def gen_predict_loader(data_opts, device, input_seqs, input_path, data_name='RNAseq', enhance_level=1):
        def set_level(level, seq_len, has_neighbor=True):
            LEN = 1000
            LEN_max = 3000
            if has_neighbor:
                if seq_len>LEN:
                    level = min(level, 3)
            else:
                if seq_len>LEN:
                    level = min(level, 2)
            if seq_len > LEN_max:
                level = 1
            return level

        BIN_WIDTH = 16
        level_dic = {0:1, 1:2, 2:4, 3:8, 4:16, 5:32}
        para_dir = os.path.dirname(os.path.abspath(args.checkpoint_dir))
        seqdata = read_yaml(os.path.join(para_dir, 'seq.data'))
        data_class = get_dataset(data_name)
        for name, info_dic in gen_info_dic(input_seqs, input_path):
            seq_len = info_dic['length']
            bin_id = seq_len // BIN_WIDTH
            has_neighbor = bin_id in seqdata
            level = set_level(enhance_level, seq_len, has_neighbor)
            sample_num = level_dic[level]-1
            if has_neighbor:
                sample_func = random.sample if len(seqdata[bin_id]) >= sample_num else random.choices
                data_opts['predict_files'] = [info_dic] + sample_func(seqdata[bin_id], k=sample_num)
                seqdata[bin_id].append(info_dic)
            else:
                data_opts['predict_files'] = [info_dic] * level_dic[level]
                seqdata[bin_id] = [info_dic]
            data_opts['Lmax'] = max(d['length'] for d in data_opts['predict_files'])
            ds = data_class(phase='predict', verbose=args.verbose, para_dir=para_dir, **data_opts)
            dl = torch.utils.data.DataLoader(ds, batch_size=len(data_opts['predict_files']), shuffle=False, drop_last=False, num_workers=1)
            yield DeviceMultiDataLoader([dl], device, keywords=ds.to_device_keywords)
    def save_pred(seq, connects, save_types, name, pred_dir, CI=None):
        ret = []
        if CI is not None:
            name = f'{name}_CI{CI:.3f}'
        for out_type in save_types:
            path = os.path.join(pred_dir, name+f'.{out_type}')
            write_SS(path, seq, connects, out_type=out_type)
            ret.append(path)
        return ret

    input_path = args.input
    input_seqs = args.seq
    output = args.output
    save_types = [args.save_type] if args.save_type!='all' else ['bpseq', 'ct', 'dbn']
    ckpt_dir = args.checkpoint_dir
    opts = get_config(args.config)
    model_name = opts['model']['model_name']

    data_name = opts['dataset']['data_name']
    data_opts = opts['dataset'][data_name]
    model_opts = opts['model'][model_name]
    common_opts = opts['common']
    data_opts.update(common_opts)
    model_opts.update(common_opts)

    # usage
    show_help_info(input_seqs, input_path)

    # prepare
    pred_dir = os.path.join(output) # save pred
    os.makedirs(pred_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # load checkpoints
    RNA_model = get_model(model_name)
    models = load_eval_checkpoints(ckpt_dir, RNA_model, model_opts, device)
    num_digit = 5
    ct = 0

    hook_dir = 'hook_features'
    hooker = None
    if args.hook_features:
        os.makedirs(hook_dir, exist_ok=True)
        hook_module_names = ['TransformerEncoderLayer', 'ResConv2dSimple']
        hooker = hook_features(models[0], hook_module_names)

    for dl in gen_predict_loader(data_opts, device, input_seqs, input_path, enhance_level=args.enhance_level):
        for data_dic, _ in dl:
            with torch.no_grad(),torch.cuda.amp.autocast():
                # torch.nan_to_num
                # BS x forward_batch_Lmax x forward_batch_Lmax
                pred_batch = torch.stack([model(data_dic) for model in models], 0).mean(0)

                # remove `begin` and `end` tokens
                forward_batch_Lmax = data_dic['forward_mask'].sum(-1).max()
                batch_Lmax = forward_batch_Lmax-2
                pred_batch = pred_batch[:, 1:batch_Lmax+1, 1:batch_Lmax+1]
                seq_onehot = data_dic['seq_onehot'][:, 1:batch_Lmax+1, :]
                nc_map = data_dic['nc_map'][:, 1:batch_Lmax+1, 1:batch_Lmax+1]
                masks = data_dic['mask'][:, 1:batch_Lmax+1, 1:batch_Lmax+1]
                seqs = data_dic['ori_seq']
                names = data_dic['name']

                if args.hook_features:
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
                    if i>0: # Notice! Only the first sample of one batch is saved.
                        break
                    ct += 1
                    length = len(seqs[i])
                    mat = pred_batch[i][masks[i]].reshape(length, length).detach().cpu().numpy()
                    mat_post = ret_pred[i][masks[i]].reshape(length, length).detach().cpu().numpy()
                    
                    CI = compute_confidence(mat, mat_post)
                    connects = arr2connects(mat_post)

                    connects = remove_lone_pairs(connects)
                    path = save_pred(seqs[i], connects, save_types, names[i], pred_dir, CI)[0]
                    seq, connects = read_SS(path)
                    CI_str = f'CI={CI:.3f}' if CI>=0.3 else 'CI<0.3'
                    print(f"[{str(ct).rjust(num_digit)}] saved in \"{path}\", {CI_str}")
                    if not args.hide_dbn:
                        print(f'{seq}\n{connects2dbn(connects)}')
                    if args.save_nc:
                        mat_nc_post = ret_pred_nc[i][masks[i]].reshape(length, length).detach().cpu().numpy()
                        
                        connects_nc = arr2connects(mat_nc_post)
                        path = save_pred(seqs[i], connects_nc, save_types, names[i]+'_nc', pred_dir, CI)[0]
                        if not args.hide_dbn:
                            seq, connects = read_SS(path)
                            print(f'{connects2dbn(connects)} NC')
    print('Finished!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seq', nargs='*', help='RNA sequences')
    parser.add_argument('-i', '--input', type=str, help='Input fasta file or directory which contains fasta files, supporting multiple seqs and multiple formats, such as fasta, bpseq, ct, dbn, or any other txet file (Only the first line will be read as input sequence).')
    parser.add_argument('-o', '--output', type=str, default='BPfold_results', help='output directory')
    parser.add_argument('--save_type', default='bpseq', choices=['bpseq', 'ct', 'dbn', 'all'], help='Saved file type.')
    parser.add_argument('--hide_dbn', action='store_true', help='Once specified, the output sequence and predicted DBN won\'t be printed.')
    parser.add_argument('-g', '--gpu', type=str, default='0')
    parser.add_argument('--checkpoint_dir', type=str, default='paras/model_predict', help='Directory of checkpionts for predicting.')
    parser.add_argument('-c', '--config', type=str, default='configs/config_predict.yaml')
    parser.add_argument('--enhance_level', type=int, default=1, choices=[0, 1, 2, 3, 4, 5],  help='Cost more time and gpu memory to possibly enhance the accuracy of output secondary structure. The bigger level, the more time and memory.')
    parser.add_argument('--save_nc', action='store_true', help='Additionally save prediction with non-canonical pairs.')
    parser.add_argument('--hook_features', action='store_true')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    seed_everything(42)
    predict(args)
