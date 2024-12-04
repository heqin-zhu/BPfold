import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class DynamicPositionBias(nn.Module):
    def __init__(self, dim, *, heads, depth, log_distance=False, norm=False):
        super().__init__()
        assert depth >= 1, 'depth for dynamic position bias MLP must be greater or equal to 1'
        self.log_distance = log_distance

        self.mlp = nn.ModuleList([])

        self.mlp.append(nn.Sequential(
            nn.Linear(1, dim),
            nn.LayerNorm(dim) if norm else nn.Identity(),
            nn.SiLU()
        ))

        for _ in range(depth - 1):
            self.mlp.append(nn.Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim) if norm else nn.Identity(),
                nn.SiLU()
            ))

        self.mlp.append(nn.Linear(dim, heads))

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, i, j):
        assert i == j
        n, device = j, self.device

        # get the (n x n) matrix of distances
        seq_arange = torch.arange(n, device = device)
        context_arange = torch.arange(n, device = device)
        indices = rearrange(seq_arange, 'i -> i 1') - rearrange(context_arange, 'j -> 1 j')
        indices += (n - 1)

        # input to continuous positions MLP
        pos = torch.arange(-n + 1, n, device = device).float() # i==j, pos.shape == (2*i-1, 1)
        pos = rearrange(pos, '... -> ... 1')

        if self.log_distance:
            pos = torch.sign(pos) * torch.log(pos.abs() + 1)  # log of distance is sign(rel_pos) * log(abs(rel_pos) + 1)

        # after applying MLP to pos,  pos.shape == (2*i-1, num_heads), i==j
        for layer in self.mlp:
            pos = layer(pos)

        # get position biases        
        bias = pos[indices]
        bias = rearrange(bias, 'i j h -> h i j') #   bias.shape == (num_heads, i, i), i==j
        return bias

def pad_at_dim(t, pad, dim = -1, value = 0.):
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

class AlibiPositionalBias(nn.Module):
    def __init__(self, heads, total_heads, **kwargs):
        super().__init__()
        self.heads = heads
        self.total_heads = total_heads

        slopes = torch.Tensor(self._get_slopes(heads))
        slopes = rearrange(slopes, 'h -> h 1 1')
        self.register_buffer('slopes', slopes, persistent = False)
        self.register_buffer('bias', None, persistent = False)
    
    def get_bias(self, i, j, device):
        i_arange = torch.arange(j - i, j, device = device)
        j_arange = torch.arange(j, device = device)
        bias = -torch.abs(rearrange(j_arange, 'j -> 1 1 j') - rearrange(i_arange, 'i -> 1 i 1'))
        return bias

    @staticmethod
    def _get_slopes(heads):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]

        if math.log2(heads).is_integer():
            return get_slopes_power_of_2(heads)

        closest_power_of_2 = 2 ** math.floor(math.log2(heads))
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:heads-closest_power_of_2]

    @property
    def device(self):
        return next(self.buffers()).device

    def forward(self, i, j):
        h, device = self.total_heads, self.device

        if self.bias is not None and self.bias.shape[-1] >= j and self.bias.shape[-2] >= i:
            return self.bias[..., -i:, -j:]

        bias = self.get_bias(i, j, device)
        bias = bias * self.slopes

        num_heads_unalibied = h - bias.shape[0]
        bias = pad_at_dim(bias, (0, num_heads_unalibied), dim = 0)
        self.register_buffer('bias', bias, persistent = False)

        return self.bias


if __name__ == '__main__':
    hidden_dim = 16
    num_heads = 3
    dynamic_pe = DynamicPositionBias(dim=hidden_dim//4, heads=num_heads, depth=2)

    alibi_heads = num_heads // 2 + (num_heads % 2 == 1)
    alibi_pe = AlibiPositionalBias(alibi_heads, num_heads)
    i, j = 128, 128
    print('dyn bias shape', dynamic_pe(i, j).shape)
    print('alibi bias shape', alibi_pe(i, j).shape)
