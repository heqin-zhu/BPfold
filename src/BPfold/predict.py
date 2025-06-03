import os, gc
import argparse
import math

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .dataset import get_dataset
from .model import get_model
from .util.misc import get_file_name, str_localtime, seed_everything
from .util.hook_features import hook_features
from .util.yaml_config import get_config, read_yaml, write_yaml
from .util.postprocess import postprocess, apply_constraints
from .util.data_sampler import DeviceMultiDataLoader
from .util.RNA_kit import read_SS, write_SS, read_fasta, connects2dbn, mat2connects, compute_confidence, remove_lone_pairs, merge_connects


SRC_DIR = os.path.dirname(os.path.realpath(__file__))


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


class BPfold_predict:
    def __init__(self, checkpoint_dir, config_file=None):
        '''
        Init

        Parameters
        ----------
        checkpoint_dir: str
            Directory of checkpoints that contain trained parameters.
        config_file: str
            if None, default: os.path.join(os.path.dirname(__file__), 'configs/config.yaml')
        '''
        self.tmp_dir = '.BPfold_tmp_files'
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

    def predict(self, input_seqs=None, input_path=None, batch_size=1, num_workers=1, hook_features=False, save_contact_map=False, save_nc=False):
        '''
        BPfold `predict function`, specify input_seqs or input_path 

        Parameters
        ----------
        input_seqs: [str]
            RNA seq list
        input_path: str
            fasta path containing multi RNA seqs or Secondary structure path in format of 'dbn', 'bpseq' or 'ct'.

        Returns
        -------
        ret: dict
            str,      str, [int],    float
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
                    connects = mat2connects(mat_post)
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
                    results = {'seq_name': seq_name, 'seq': seqs[i], 'connects': connects, 'CI': CI}

                    ## NC pairs
                    if save_nc: # not accurate, to be improved
                        mat_nc_post = ret_pred_nc[i][masks[i]].reshape(length, length).detach().cpu().numpy()
                        results['connects_nc'] = mat2connects(mat_nc_post)

                    yield results

                # clean memory
                del data_dic, pred_batch, seq_onehot, nc_map, masks
                gc.collect()
                torch.cuda.empty_cache()


    def gen_info_dic(self, input_seqs, input_path, data_name='RNAseq'):
        def valid_seq(seq):
            return seq.isalpha()
        def process_one_seq(name, seq, src_fasta_path=''):
            if valid_seq(seq):
                return {'seq': seq, 'name': name, 'length': len(seq), 'dataset': data_name}
            else:
                print(f'[Warning] Invalid seq, containing non-alphabet character, ignored: {src_fasta_path}{name}="{seq}"')
        def process_one_file(file_path, data_name='RNAseq'):
            file_name, suf = get_file_name(file_path, return_suf=True)
            if suf.lower() in {'.fasta', '.fa'}: # fasta file
                for name, seq in read_fasta(file_path):
                    info_dic = process_one_seq(name, seq, file_path+': ')
                    if info_dic is not None:
                        yield info_dic
            elif suf.lower() in {'.dbn', '.ct', '.bpseq'}: # SS file
                seq, _ = read_SS(file_path)
                yield {'path': file_path, 'length': len(seq), 'dataset': data_name}
            else: # unknown filetype
                print(f'[Warning] Unknown file type, ignored: {file_path}')

        if input_seqs:
            if isinstance(input_seqs, str):
                input_seqs = [input_seqs]
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

    def save_pred_results(self, pred_results, save_dir='.', save_name=None, out_type='csv', hide_dbn=False):
        '''
        Save pred_results predicted by `BPfold_predict.predict` in `save_dir` in format of `out_type`.

        Parameters
        ----------
        pred_results: Iterable(dict)
            Each item is a dict which contains keys `seq_name`, `seq`, `connects` and `CI` (optional: connects_nc).
        save_dir: str
            save_dir, if not exist, will mkdir.
        save_name: str
            if not available, use basename of save_dir
        out_type: str
            csv, bpseq, ct, dbn
        '''
        def print_result(save_name, idx, seq, dbn, CI, hide_dbn=False, num_digit=7):
            CI_str = f'CI={CI:.3f}' if CI>=0.3 else 'CI<0.3'
            print(f"[{str(idx).rjust(num_digit)}] saved in \"{save_name}\", {CI_str}")
            if not hide_dbn:
                print(f'{seq}\n{dbn}')

        os.makedirs(save_dir, exist_ok=True)
        if out_type=='csv':
            if save_name is None:
                save_name = os.path.basename(os.path.abspath(save_dir))
            df = pd.DataFrame(pred_results)
            df['dbn'] = df['connects'].apply(connects2dbn)
            # df_out = df[['seq_name', 'seq', 'dbn', 'CI']]
            df_out = df
            csv_path = os.path.join(save_dir, f'{save_name}.csv')
            num_digit = math.ceil(math.log(len(df_out), 10))
            for idx, row in enumerate(df_out.itertuples()):
                print_result(f'{csv_path}:line{idx+2}:{row.seq_name}', idx+1, row.seq, row.dbn, row.CI, hide_dbn=hide_dbn, num_digit=num_digit)
            df_out.to_csv(csv_path, index=False)
            print(f"Predicted structures in format of dot-bracket are saved in \"{csv_path}\".")

            ## NC pairs
            if 'connects_nc' in df.columns: # not accurate, to be improved
                df['dbn_nc'] = df['connects_nc'].apply(connects2dbn)
                df_out_nc = df[['seq_name', 'seq', 'dbn', 'dbn_nc', 'CI']]
                csv_path_nc = os.path.join(save_dir, f'{save_name}_nc.csv')
                df_out_nc.to_csv(csv_path_nc, index=False)
        else:
            num_digit = 7
            confidence_dic = {}
            for ct, res_dic in enumerate(pred_results):
                seq_name = res_dic['seq_name']
                seq = res_dic['seq']
                connects = res_dic['connects']
                CI = res_dic['CI']
                confidence_dic[seq_name] = CI
                path = os.path.join(save_dir, seq_name+f'.{out_type}')
                write_SS(path, seq, connects)
                print_result(path, ct+1, seq, connects2dbn(connects), CI, hide_dbn=hide_dbn, num_digit=num_digit)

                ## NC pairs
                if 'connects_nc' in res_dic: # not accurate, to be improved
                    connects_nc = res_dic['connects_nc']
                    path = os.path.join(save_dir, seq_name+'_nc'+f'.{out_type}')
                    write_SS(path, seq, connects_nc)
                    connects_mix = merge_connects(connects, connects_nc)
                    path = os.path.join(save_dir, seq_name+'_mix'+f'.{out_type}')
                    write_SS(path, seq, connects_mix)
                    if not hide_dbn:
                        print(f'{connects2dbn(connects_nc)} NC')
                        print(f'{connects2dbn(connects_mix)} MIX')
            confidence_path = os.path.join(os.path.dirname(save_dir), os.path.basename(save_dir)+f'_confidence_{str_localtime()}.yaml')
            if confidence_dic:
                print(f"Confidence indexes are saved in \"{confidence_path}\"")
                write_yaml(confidence_path, confidence_dic)


def show_examples():
    method_name = 'BPfold'
    print('Please specify "--seq" or "--input" argument for input sequences or input file. Such as:')
    print(f'$ {method_name} --checkpoint_dir PATH_TO_CHECKPOINT --seq GGUAAAACAGCCUGU AGUAGGAUGUAUAUG --output {method_name}_results')
    print(f'$ {method_name} --checkpoint_dir PATH_TO_CHECKPOINT --input examples/examples.fasta # (multiple sequences are supported)')
    print(f'$ {method_name} --checkpoint_dir PATH_TO_CHECKPOINT --input examples/URS0000D6831E_12908_1-117.bpseq # .bpseq, .ct, .dbn')
    exit()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint_dir', type=str, help='Directory of checkpoints that contain trained parameters.', required=True)
    parser.add_argument('-s', '--seq', nargs='*', help='RNA sequences')
    parser.add_argument('-i', '--input', type=str, help='Input fasta file or directory which contains fasta files, supporting multiple seqs and multiple formats, such as fasta, bpseq, ct or dbn.')
    parser.add_argument('-o', '--output', type=str, default='BPfold_results', help='output directory')
    parser.add_argument('-g', '--gpu', type=str, default='0')
    parser.add_argument('--out_type', default='csv', choices=['bpseq', 'ct', 'dbn', 'csv'], help='Saved file type.')
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

    # usage
    if args.input is None and args.seq is None:
        show_examples()

    BPfold_predictor = BPfold_predict(checkpoint_dir=args.checkpoint_dir)
    pred_results = BPfold_predictor.predict(args.seq, args.input, args.batch_size, args.num_workers, args.hook_features, args.save_contact_map, save_nc=args.save_nc)
    save_name = get_file_name(args.input) if args.input else None
    BPfold_predictor.save_pred_results(pred_results, save_dir=args.output, save_name=save_name, out_type=args.out_type)
    del BPfold_predictor
    gc.collect()
    torch.cuda.empty_cache()
    print('Program Finished!')


if __name__ == '__main__':
    main()
