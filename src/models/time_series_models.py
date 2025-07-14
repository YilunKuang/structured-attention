import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from src.model import *

@dataclass
class GPTforTimeSeriesConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    d_head: int = 64
    d_qk_head: int = 64
    d_model: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    split_qkv: bool = False 
    do_qk_ln: bool = False
    manual_disable_flash_att: bool = False
    token_mixing_struct: str = "low_rank"
    link_function: str = "softmax"

    # Causal Mask
    causal: bool = True

    # Sliding Window Attention
    sliding_block_size: int = 1024
    use_swa_with_for_loop: bool = False

    # SWA + Global (GSWA)
    gswa_rank_list: str = "32|32"
    
    # Sequential GSWA
    sequential_GSWA_layer_list: str = "MHA|SWA|SWA|MHA|SWA|SWA"
    
    # Multi-Head Attention
    mha_SP_attn_logits_scaling: bool = False

    # MLR
    mlr_rank_list: str = "16|16|16|16"
    mlr_block_size_list: str = "default"
    mlr_divide_by_num_levels: bool = False
    mlr_block_divide_by_num_levels: bool = False

    # BilinearMLR
    bilinear_mlr_muP_attn_logits_scaling: bool = False

    # debug flag
    debug: bool = False

class GPTforTimeSeries(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None

        if config.d_head == -1:
            assert config.d_model % config.n_head == 0, "d_model must be divisible by n_head"
            config.d_head = config.d_model // config.n_head
        elif config.n_head == -1:
            assert config.d_model % config.d_head == 0, "d_model must be divisible by d_head"
            config.n_head = config.d_model // config.d_head
        else:
            assert config.d_model == config.d_head * config.n_head
        
        self.config = config
        self.block_size = config.block_size

        self._read_in = nn.Linear(1, config.d_model, bias=config.bias)
        self._backbone = nn.ModuleDict(dict(
            wpe = nn.Embedding(config.block_size, config.d_model),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config, i_layer) for i_layer in range(config.n_layer)]),
            ln_f = LayerNorm(config.d_model, bias=config.bias),
        ))
        self._read_out = nn.Linear(config.d_model, 1, bias=config.bias)

        # init all weights
        self.apply(self._init_weights)

        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, x, y=None):
        x = self._read_in(x)
        x = self._backbone.drop(x)
        for block in self._backbone.h:
            x = block(x)
        x = self._backbone.ln_f(x)
        x = self._read_out(x)

        return x

        # if y==None: 
        #     raise NotImplementedError
        # else:


        # breakpoint()


        # # x = self.embedding(x)
        # # x = self.transformer(x)
        # # return self.fc(x[:, -1, :])

        # if inds is None:
        #     inds = torch.arange(ys.shape[1])
        # else:
        #     inds = torch.tensor(inds)
        #     if max(inds) >= ys.shape[1] or min(inds) < 0:
        #         raise ValueError("inds contain indices where xs and ys are not defined")
        # zs = self._combine(xs, ys)
        # embeds = self._read_in(zs)

        # # backbone computation
        # device = embeds.device
        # b, t, c = embeds.size()
        # assert t <= (self.block_size), f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        # pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # # forward the GPT model itself
        # pos_emb = self._backbone.wpe(pos) # position embeddings of shape (t, d_model)
        # output = self._backbone.drop(embeds + pos_emb)
        # for block in self._backbone.h:
        #     output = block(output)
        # output = self._backbone.ln_f(output)

        # # output = self._backbone(inputs_embeds=embeds) #.last_hidden_state
        # prediction = self._read_out(output)
        # return prediction[:, ::2, 0][:, inds]  # predict only on xs

