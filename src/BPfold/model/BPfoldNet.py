from typing import ClassVar

import numpy as np
import torch
import torch.nn as nn

from .pos_embed import *


class SE_Block(nn.Module):
    "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"
    def __init__(self, c, r=1):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)          
        
class ResConv2dSimple(nn.Module):
    def __init__(self, 
                 in_c, 
                 out_c,
                 kernel_size=7,
                 use_se = False,
                ):  
        super().__init__()
        if use_se:
            self.conv = nn.Sequential(
                # b c w h
                nn.Conv2d(in_c,
                          out_c, 
                          kernel_size=kernel_size, 
                          padding="same", 
                          bias=False),
                # b w h c#
                nn.BatchNorm2d(out_c),
                SE_Block(out_c),
                nn.GELU(),
                # b c e 
            )
            
        else:
            self.conv = nn.Sequential(
                # b c w h
                nn.Conv2d(in_c,
                          out_c, 
                          kernel_size=kernel_size, 
                          padding="same", 
                          bias=False),
                # b w h c#
                nn.BatchNorm2d(out_c),
                nn.GELU(),
                # b c e 
            )
        
        if in_c == out_c:
            self.res = nn.Identity()
        else:
            self.res = nn.Sequential(
                nn.Conv2d(in_c,
                          out_c, 
                          kernel_size=1, 
                          bias=False)
            )

    def forward(self, x):
        # b e s 
        h = self.conv(x)
        x = self.res(x) + h
        return x
    

class MultiHeadSelfAttention(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 positional_embedding: str,
                 num_heads: int = None,
                # k_dim: int = None,
                # v_dim: int = None,
                 dropout: float = 0.10, 
                 bias: bool = True,
                 temperature: float = 1,
                 use_se = False,
                ):
        super().__init__()
        
        assert positional_embedding in ("dyn", "alibi")
        self.positional_embedding = positional_embedding
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads or 1
        self.head_size = hidden_dim//self.num_heads
        assert hidden_dim == self.head_size*self.num_heads, "hidden_dim must be divisible by num_heads"
        self.dropout = dropout
        self.bias = bias
        self.temperature = temperature
        self.use_se = use_se
        
        if self.positional_embedding == "dyn":
            self.dynpos = DynamicPositionBias(dim = hidden_dim//4, heads = num_heads, depth = 2)
        elif self.positional_embedding == "alibi":
            alibi_heads = num_heads // 2 + (num_heads % 2 == 1)
            self.alibi = AlibiPositionalBias(alibi_heads, self.num_heads)
            
        self.dropout_layer = nn.Dropout(dropout)
        self.weights = nn.Parameter(torch.empty(self.hidden_dim, 3 * self.hidden_dim)) # QKV
        self.out_w = nn.Parameter(torch.empty(self.hidden_dim, self.hidden_dim)) # QKV
        torch.nn.init.xavier_normal_(self.weights)
        torch.nn.init.xavier_normal_(self.out_w)

        if self.bias:
            self.out_bias = nn.Parameter(torch.empty(1, 1, self.hidden_dim)) # QKV
            self.in_bias = nn.Parameter(torch.empty(1, 1, 3*self.hidden_dim)) # QKV
            torch.nn.init.constant_(self.out_bias, 0.)
            torch.nn.init.constant_(self.in_bias, 0.)
        if not use_se:
            self.gamma = nn.Parameter(torch.ones(self.num_heads).view(1, -1, 1, 1))

    def forward(self, x, adj=None, mask=None, same=True, return_attn_weights=False):
        '''
            x: sequence feature: BxLxD
            adj: adj_matrix feature: Bx1xLxL
        '''
        b, l, h = x.shape
        x = x @ self.weights # b, l, 3*hidden
        if self.bias:
            x = x + self.in_bias
        Q, K, V = x.view(b, l, self.num_heads, -1).permute(0,2,1,3).chunk(3, dim=3) # b, a, l, head
        
        norm = self.head_size**0.5
        attention = (Q @ K.transpose(2,3)/self.temperature/norm)
        
        if self.positional_embedding == "dyn":
            i, j = map(lambda t: t.shape[-2], (Q, K))
            attn_bias = self.dynpos(i, j).unsqueeze(0)
            attention = attention + attn_bias
        elif self.positional_embedding == "alibi":
            i, j = map(lambda t: t.shape[-2], (Q, K))
            attn_bias = self.alibi(i, j).unsqueeze(0)
            attention = attention + attn_bias
        elif self.positional_embedding == "xpos":
            self.xpos = XPOS(self.head_size)

        if adj is not None:
            if not self.use_se:
                adj = self.gamma * adj
            attention = attention + adj

        attention = attention.softmax(dim=-1) # b, a, l, l, softmax won't change shape
        if mask is not None:
            ## (batch_size, seq_len)->(batch_size, seq_len, seq_len) -> (batch_size, num_heads, seq_len, seq_len)
            ## `&` op can be replaced by torch.bmm, more efficient for large computation. To Fix: "baddbmm_cuda" not implemented for 'Bool'
            mask2d = (mask.unsqueeze(1) & mask.unsqueeze(2)).unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            # mask2d = mask.view(b,1,1,-1).repeat(1, self.num_heads, l, 1)  # wrong when broadcasting, reference for reproducing paper data
            attention = attention*mask2d
        '''
        ### Note: Different batch sizes result in slightly different outputs because of padding.
        ### There are many calculations that are influenced by padding: 
        1. another way for cal attetion:  attention = attention.masked_fill(~mask2d, float('-inf') => attention.softmax(dim=-1) =>  attention *= mask2d
        2. the out_bias may be masked: out_bias = out_bias*mask.unsqueeze(-1)
        3. the adj when apply conv: adj = adj*mask2d
        Since the differences are small, we ignore it.
        '''

        out = attention @ V  # b, a, l, head
        out = out.permute(0,2,1,3).flatten(2,3) # b, a, l, head -> b, l, (a, head) -> b, l, hidden
        if self.bias:
            out = out + self.out_bias
        if return_attn_weights:
            return out, attention
        else:
            return out           


class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 positional_embedding: str,
                 num_heads: int = None,
                 dropout: float = 0.10,
                 ffn_size: int = None,
                 activation: nn.Module = nn.GELU,
                 temperature: float = 1.,
                 use_se = False,
                ):
        super().__init__()
        if num_heads is None:
            num_heads = 1
        if ffn_size is None:
            ffn_size = hidden_dim*4
        self.in_norm = nn.LayerNorm(hidden_dim)
        self.mhsa = MultiHeadSelfAttention(hidden_dim=hidden_dim,
                                           num_heads=num_heads,
                                           positional_embedding=positional_embedding,
                                           dropout=dropout,
                                           bias=True,
                                           temperature=temperature,
                                           use_se=use_se,
                                          )
        self.dropout_layer = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, ffn_size),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(ffn_size, hidden_dim),
            nn.Dropout(dropout)
        )
        

    def forward(self, x, adj=None, mask = None, return_attn_weights = False):
        '''
            x: sequence feature: BxLXD
            adj: adj_matrix feature: Bx1xLxL
        '''
        x_in = x
        if return_attn_weights:
            x, attn_w = self.mhsa(self.in_norm(x), adj=adj, mask=mask, return_attn_weights = True)
        else:
            x = self.mhsa(self.in_norm(x), adj=adj, mask=mask, return_attn_weights = False)
        x = self.dropout_layer(x) + x_in
        x = self.ffn(x) + x

        if return_attn_weights:
            return x, attn_w
        else:
            return x
        

class AdjTransformerEncoder(nn.Module):
    def __init__(self,
                 positional_embedding: str,
                 dim: int  = 192,
                 head_size: int = 32,
                 dropout: float = 0.10,
                 dim_feedforward: int = 192 * 4,
                 activation: nn.Module = nn.GELU,
                 temperature: float = 1.,
                 num_layers: int = 12,
                 num_adj_convs: int =3,
                 ks: int = 3,
                 use_se = False,
                 conv_in_chan=0,
                ):
        super().__init__()
        num_heads, rest = divmod(dim, head_size)
        assert rest == 0
        self.num_heads = num_heads
        
        self.layers = nn.Sequential(
            *[TransformerEncoderLayer(hidden_dim=dim,
                                     num_heads=num_heads,
                                     positional_embedding=positional_embedding,
                                     dropout=dropout,
                                     ffn_size=dim_feedforward,
                                     activation=activation,
                                     temperature=temperature,
                                     use_se=use_se,
                                    ) 
             for i in range(num_layers)]
        )
        self.conv_in_chan = conv_in_chan
        if conv_in_chan>0:
            self.conv_layers = nn.ModuleList()
            for i in range(num_adj_convs):
                self.conv_layers.append(ResConv2dSimple(in_c=conv_in_chan if i == 0 else num_heads, out_c=num_heads, kernel_size=ks, use_se=use_se))
            
            
    def forward(self, x, adj, mask=None):
        '''
            x: sequence feature: BxLxD
            adj: adj_matrix feature: Bx1xLxL
        '''
        for ind, mod in enumerate(self.layers):
            if self.conv_in_chan>0:
                if ind < len(self.conv_layers):
                    adj = self.conv_layers[ind](adj)
            x = mod(x, adj=adj, mask=mask)
        return x

        
class BPfold_model(nn.Module):
    def __init__(self,  
                 positional_embedding: str,
                 not_slice: bool,
                 num_convs,
                 adj_ks=3,
                 dim=192, 
                 depth=12,
                 head_size=32,
                 use_se=False,
                 embed_filter=False,
                 use_BPP=True,
                 use_BPE=True,
                 dispart_outer_inner=True,
                 *args,
                 **kargs,
                 ):
        super().__init__()
        self.slice_tokens = not not_slice  # removing unnecessary padding tokens
        if num_convs is None:
            num_convs = depth
        # 4 nucleotides + 3 tokens: AUGC, BEGIN, END, EMPTY
        self.emb = nn.Embedding(4+3, dim)


        conv_in_chan = 0
        if use_BPP:
            conv_in_chan += 1
        if use_BPE:
            conv_in_chan += 1 + dispart_outer_inner

        self.transformer = AdjTransformerEncoder(
            num_layers=depth,
            num_adj_convs=num_convs,
            dim=dim,
            head_size=head_size,
            positional_embedding=positional_embedding,
            ks=adj_ks,
            use_se=use_se,
            conv_in_chan=conv_in_chan,
        )
        
        # self.proj_out = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 2))
        
        self.struct_embeds = nn.ModuleDict()
        
        self.use_BPP = use_BPP
        self.use_BPE = use_BPE
        self.embed_filter = embed_filter
        if self.embed_filter:
            self.filter_embedding = nn.Embedding(2, dim)
            
    def forward(self, x0):
        forward_mask = x0['forward_mask']
        x = x0['input'] # BxL
        mat = None
        if self.slice_tokens:
            forward_batch_Lmax = forward_mask.sum(-1).max()
            forward_mask = forward_mask[:,:forward_batch_Lmax]
            x = self.emb(x[:, :forward_batch_Lmax]) # x: B x forward_batch_Lmax x D
            if self.use_BPP:
                BPPM = x0['BPPM'][:, :, :forward_batch_Lmax, :forward_batch_Lmax] # Bx1xLxL -> B x 1 x batch_Lmax x batch_Lmax
                mat = torch.cat([mat, BPPM], dim=1) if mat is not None else BPPM
            if self.use_BPE:
                BPEM = x0['BPEM'][:, :, :forward_batch_Lmax, :forward_batch_Lmax] # Bx1xLxL -> B x 1 x batch_Lmax x batch_Lmax
                mat = torch.cat([mat, BPEM], dim=1) if mat is not None else BPEM
        if self.embed_filter:
            filter_embeded = self.filter_embedding(x0['is_good']).unsqueeze(1) # Bx1xE
            x = x + filter_embed
        
        x = self.transformer(x, mat, mask=forward_mask) # B x forward_batch_Lmax x D
        out = x @ x.transpose(-1, -2) # B x forward_batch_Lmax x forward_batch_Lmax
        return out


if __name__ == '__main__':
    arg_dic = {
               'adj_ks': 3, # int
               'depth': 12,
               'dim': 256,
               'head_size': 32,
               'not_slice': False,
               'num_convs': 3,
               'positional_embedding': 'dyn', # alibi
               'use_se': True,
              }
    Lmax = 128
    batch_size = 32
    x0 = {
          'forward_mask': torch.ones(batch_size, Lmax, dtype=bool),
          'input': torch.randint(0, 7, (batch_size, Lmax)),
          'BPEM': torch.randn((batch_size, 1, Lmax, Lmax)),
          'BPPM': torch.randn((batch_size, 1, Lmax, Lmax)),
         }
    model =  BPfold_model(**arg_dic)
    out = model(x0)
    print(model)
    print('out', out.shape)
    # for name, para in model.named_parameters():
    #     print(name)
