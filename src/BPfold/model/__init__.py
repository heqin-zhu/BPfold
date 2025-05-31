from .BPfoldNet import BPfold_model

def get_model(s):
    return {
            'bpfold': BPfold_model,
            'bpfold_model': BPfold_model,
           }[s.lower()]
