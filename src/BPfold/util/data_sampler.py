import torch


class LenMatchBatchSampler(torch.utils.data.BatchSampler):
    def __iter__(self):
        buckets = {}
        yielded = 0

        for idx in self.sampler:
            s = self.sampler.data_source[idx]
            if isinstance(s, tuple): 
                L = s[0]["length"]
            else: 
                L = s["length"]
            L = max(1, L // 16) 
            if L not in buckets:  
                buckets[L] = []
            buckets[L].append(idx)
            
            if len(buckets[L]) == self.batch_size:
                batch = list(buckets[L])
                yield batch
                yielded += 1
                buckets[L] = []
                
        batch = []
        leftover = [idx for bucket in buckets.values() for idx in bucket]

        for idx in leftover:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yielded += 1
                yield batch
                batch = []

        if len(batch) > 0 and not self.drop_last:
            yielded += 1
            yield batch


class DeviceMultiDataLoader:
    def __init__(self, dataloader_list, device='cuda', keywords=None):
        if not isinstance(dataloader_list, list):
            raise Exception('[Error] List of dataloaders expected.')
        self.device = device
        self.keywords = keywords
        self.dataloader_list = dataloader_list
        self.length = sum([len(l) for l in self.dataloader_list])
    
    def __len__(self):
        return self.length
    
    def __iter__(self):
        for dataloader in self.dataloader_list:
            for batch in dataloader:
                yield tuple({k: x[k].to(self.device) if self.keywords is None or k in self.keywords else x[k] for k in x} for x in batch)
