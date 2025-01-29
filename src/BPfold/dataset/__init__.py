from .RNAseq import RNAseq_data


def get_dataset(s):
    return {
            'rnaseq': RNAseq_data,
           }[s.lower()]
