"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from sympy import factorint, divisors
from einops import rearrange

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_head == 0

        self.split_qkv = config.split_qkv 
        self.do_qk_ln = config.do_qk_ln
        self.manual_disable_flash_att = config.manual_disable_flash_att
        self.link_function = config.link_function
        self.mha_SP_attn_logits_scaling = config.mha_SP_attn_logits_scaling

        if self.do_qk_ln:
            self.ln_q = LayerNorm(config.d_qk_head, bias=config.bias)
            self.ln_k = LayerNorm(config.d_qk_head, bias=config.bias)

        if self.split_qkv:
            self.c_attn_q = nn.Linear(config.d_model, config.d_qk_head * config.n_head, bias=config.bias)
            self.c_attn_k = nn.Linear(config.d_model, config.d_qk_head * config.n_head, bias=config.bias)
            self.c_attn_v = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        else:
            self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
    
        # output projection
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.d_model = config.d_model
        self.d_qk_head = config.d_qk_head
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        if self.manual_disable_flash_att:
            self.flash = False
        
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (d_model)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        if self.split_qkv:
            q = self.c_attn_q(x)
            k = self.c_attn_k(x)
            v = self.c_attn_v(x)
        else:
            q, k, v  = self.c_attn(x).split(self.d_model, dim=2)
        
        k = k.view(B, T, self.n_head, self.d_qk_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, self.d_qk_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # QK-layer normalization (per-head)
        if self.do_qk_ln:
            q = self.ln_q(q)
            k = self.ln_k(k)

        # muP attention
        if self.mha_SP_attn_logits_scaling:
            pass
        else:
            k = k * (8 / math.sqrt(k.size(-1)))

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

            if self.link_function == "softmax":
                att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
                att = F.softmax(att, dim=-1)
            elif self.link_function == "identity":
                att = att.masked_fill(self.bias[:, :, :T, :T] == 0, 0)
            else:
                raise ValueError 
            
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class CausalSelfSlidingWindowAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_head == 0
        
        self.split_qkv = config.split_qkv 
        self.do_qk_ln = config.do_qk_ln
        self.manual_disable_flash_att = config.manual_disable_flash_att
        self.link_function = config.link_function
        self.mha_SP_attn_logits_scaling = config.mha_SP_attn_logits_scaling
        self.sliding_block_size = config.sliding_block_size
        self.use_swa_with_for_loop = config.use_swa_with_for_loop

        if self.do_qk_ln:
            self.ln_q = LayerNorm(config.d_qk_head, bias=config.bias)
            self.ln_k = LayerNorm(config.d_qk_head, bias=config.bias)

        if self.split_qkv:
            self.c_attn_q = nn.Linear(config.d_model, config.d_qk_head * config.n_head, bias=config.bias)
            self.c_attn_k = nn.Linear(config.d_model, config.d_qk_head * config.n_head, bias=config.bias)
            self.c_attn_v = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        else:
            self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
    
        # output projection
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.d_model = config.d_model
        self.d_qk_head = config.d_qk_head
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        assert self.manual_disable_flash_att==True
        if self.manual_disable_flash_att:
            self.flash = False
        
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.triu(torch.tril(torch.ones(config.block_size, config.block_size)), diagonal=-self.sliding_block_size)
                                        .view(1, 1, config.block_size, config.block_size))

    def sliding_window_attn_optimal_FLOPs(self, q, k):
        B, H, T, d_head = q.shape

        att = torch.zeros(B, H, T, T, device=q.device, dtype=q.dtype)

        for i in range(T):
            q_i = q[:, :, i:i+1, :]  # Shape: (B, H, 1, d_head)

            start = max(0, i - self.sliding_block_size)
            end = min(T, i + self.sliding_block_size)

            k_window = k[:, :, start:end, :]
            qk_score_at_i = torch.matmul(q_i, k_window.transpose(-1, -2))
            
            att[:, :, i:i+1, start:end] = qk_score_at_i

        return att
    
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (d_model)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        if self.split_qkv:
            q = self.c_attn_q(x)
            k = self.c_attn_k(x)
            v = self.c_attn_v(x)
        else:
            q, k, v  = self.c_attn(x).split(self.d_model, dim=2)
        
        k = k.view(B, T, self.n_head, self.d_qk_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, self.d_qk_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # QK-layer normalization (per-head)
        if self.do_qk_ln:
            q = self.ln_q(q)
            k = self.ln_k(k)

        # muP attention
        if self.mha_SP_attn_logits_scaling:
            pass
        else:
            k = k * (8 / math.sqrt(k.size(-1)))

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            if self.use_swa_with_for_loop:
                att = self.sliding_window_attn_optimal_FLOPs(q, k) * (1.0 / math.sqrt(k.size(-1)))
            else:
                att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

            if self.link_function == "softmax":
                att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
                att = F.softmax(att, dim=-1)
            elif self.link_function == "identity":
                att = att.masked_fill(self.bias[:, :, :T, :T] == 0, 0)
            else:
                raise ValueError 
            
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class CausalSelfGlobalSlidingWindowAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_head == 0
        assert config.split_qkv == True
        assert config.do_qk_ln == True
        assert config.mha_SP_attn_logits_scaling==False
        assert config.link_function == "softmax"
        
        self.split_qkv = config.split_qkv 
        self.do_qk_ln = config.do_qk_ln
        self.manual_disable_flash_att = config.manual_disable_flash_att
        self.link_function = config.link_function
        self.mha_SP_attn_logits_scaling = config.mha_SP_attn_logits_scaling
        self.sliding_block_size = config.sliding_block_size
        self.use_swa_with_for_loop = config.use_swa_with_for_loop
        self.gswa_rank_list = list(map(int, config.gswa_rank_list.split("|")))
        assert len(self.gswa_rank_list)==2, "only support a|b, where a is global d_head and b is local d_head"

        # QK LayerNorm
        self.ln_q = LayerNorm(config.d_qk_head, bias=config.bias)
        self.ln_k = LayerNorm(config.d_qk_head, bias=config.bias)

        # Q, K, V
        self.c_attn_q = nn.Linear(config.d_model, config.d_qk_head * config.n_head, bias=config.bias)
        self.c_attn_k = nn.Linear(config.d_model, config.d_qk_head * config.n_head, bias=config.bias)
        self.c_attn_v = nn.Linear(config.d_model, config.d_model, bias=config.bias)
    
        # output projection
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.d_model = config.d_model
        self.d_qk_head = config.d_qk_head
        self.dropout = config.dropout
        self.block_size = config.block_size
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        assert self.manual_disable_flash_att==True
        self.flash = False
        
        print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias_global", torch.tril(torch.ones(config.block_size, config.block_size))
                                    .view(1, 1, config.block_size, config.block_size))

        self.register_buffer("bias_SWA", torch.triu(torch.tril(torch.ones(config.block_size, config.block_size)), diagonal=-self.sliding_block_size)
                                    .view(1, 1, config.block_size, config.block_size))

    def sliding_window_attn_optimal_FLOPs(self, q, k):
        B, H, T, d_head = q.shape

        att = torch.zeros(B, H, T, T, device=q.device, dtype=q.dtype)

        for i in range(T):
            q_i = q[:, :, i:i+1, :]  # Shape: (B, H, 1, d_head)

            start = max(0, i - self.sliding_block_size)
            end = min(T, i + self.sliding_block_size)

            k_window = k[:, :, start:end, :]
            qk_score_at_i = torch.matmul(q_i, k_window.transpose(-1, -2))
            
            att[:, :, i:i+1, start:end] = qk_score_at_i

        return att
    
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (d_model)
        assert self.block_size==T, "didn't implement inference"

        # compute Q, K, V
        q = self.c_attn_q(x)
        k = self.c_attn_k(x)
        v = self.c_attn_v(x)

        q = q.view(B, T, self.n_head, self.d_qk_head).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, self.d_qk_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # QK LayerNorm
        q = self.ln_q(q)
        k = self.ln_k(k)
        
        # split d_head based on "a|b"
        q_tuples = torch.split(q, self.gswa_rank_list, dim=-1)
        k_tuples = torch.split(k, self.gswa_rank_list, dim=-1)

        # Global Compute
        att = (q_tuples[0] @ k_tuples[0].transpose(-2, -1)) * (8.0 / self.gswa_rank_list[0])
        att = att.masked_fill(self.bias_global[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        # Local Compute
        if self.use_swa_with_for_loop:
            att_local = self.sliding_window_attn_optimal_FLOPs(q_tuples[1], k_tuples[1]) * (8.0 / self.gswa_rank_list[1])
        else:
            att_local = (q_tuples[1] @ k_tuples[1].transpose(-2, -1)) * (8.0 / self.gswa_rank_list[1])

        att_local = att_local.masked_fill(self.bias_SWA[:,:,:T,:T] == 0, float('-inf'))
        att_local = F.softmax(att_local, dim=-1)

        # Combine Global and Local Compute
        att = att + att_local

        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class BilinearBTT_W_K_Projection(nn.Module):
    def __init__(self, shape):
        super().__init__()
        # self.weight.shape = (c, d, Hbs)
        self.weight = nn.Parameter(torch.randn(shape))

    def forward(self, x):
        '''x.shape = (c, BT, d)'''

        if True:
            RMS_W_K = torch.sqrt(torch.mean(self.weight**2.) + 1e-8)
            d_in = self.weight.size(-2)
            d_out = self.weight.size(-1)
            init_scale_W_K = (min(d_in, d_out) / (d_in * d_in))**0.5
            W_K_normed = self.weight / max(1, RMS_W_K / init_scale_W_K)

        # output.shape = (c, BT, Hbs) = (c, BT, d) * (c, d, Hbs)
        return torch.bmm(x, W_K_normed)

class BilinearBTT_W_Q_Projection(nn.Module):
    def __init__(self, shape):
        super().__init__()
        # self.weight.shape = (Hb, a, cs)
        self.weight = nn.Parameter(torch.randn(shape))

    def forward(self, x):
        '''x.shape = (Hb, cs, BT)'''

        if True:
            RMS_W_Q = torch.sqrt(torch.mean(self.weight**2.) + 1e-8)
            d_in = self.weight.size(-1)
            d_out = self.weight.size(-2)
            init_scale_W_Q = (min(d_in, d_out) / (d_in * d_in))**0.5
            W_Q_normed = self.weight / max(1, RMS_W_Q / init_scale_W_Q)

        # output.shape = (Hb, a, BT) = (Hb, a, cs) * (Hb, cs, BT)
        return torch.bmm(W_Q_normed, x)


class BilinearBTTAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_head == 0
        assert config.d_model == config.d_qk_head * config.n_head
        assert config.split_qkv==True
        assert config.manual_disable_flash_att==True
        assert config.do_qk_ln==True

        self.split_qkv = config.split_qkv 
        self.do_qk_ln = config.do_qk_ln
        self.manual_disable_flash_att = config.manual_disable_flash_att
        self.link_function = config.link_function

        self.bilinear_btt_muP_attn_logits_scaling = config.bilinear_btt_muP_attn_logits_scaling
        self.bilinearBTT_use_extra_LN_on_X = config.bilinearBTT_use_extra_LN_on_X

        self.n_head = config.n_head
        self.d_model = config.d_model
        self.d_qk_head = config.d_qk_head
        self.dropout = config.dropout

        self.btt_tt_dim = config.btt_tt_dim
        self.btt_tt_rank = config.btt_tt_rank

        # define BTT shape
        self.a, self.b = self.c, self.d = self.factorize(self.d_model, self.btt_tt_dim)
        self.s = self.btt_tt_rank

        # define BTT projection matrices
        self.bilinear_btt_wk = BilinearBTT_W_K_Projection((self.c, self.d, self.n_head*self.b*self.s))
        self.bilinear_btt_wq = BilinearBTT_W_Q_Projection((self.n_head*self.b, self.a, self.c*self.s))

        if self.do_qk_ln:
            self.bilinear_btt_ln_PLPRXT = LayerNorm(config.d_model, bias=config.bias)
        if self.bilinearBTT_use_extra_LN_on_X:
            self.bilinear_btt_ln_X = LayerNorm(config.d_model, bias=config.bias)
        
        # W_V, W_O Projections
        self.c_attn_v = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.causal = (not hasattr(config, "causal")) or (config.causal == True)
        if self.causal:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def factorize(self, x, n=2):
        if n == 2:
            bigger = next(factor for factor in divisors(x) if factor > math.sqrt(x))
            return [x//bigger, bigger]

        # Get prime factors and their counts
        prime_factors = factorint(x)

        # Initialize the n integers
        numbers = [1] * n

        # Distribute the prime factors
        for prime, count in reversed(list(prime_factors.items())):
            for _ in range(count):
                # Find the number with the smallest product to assign the prime factor
                min_index = min(range(n), key=lambda i: numbers[i])
                numbers[min_index] *= prime

        # return in ascending order
        return sorted(numbers)

    def bilinear_btt_einsum(self, x):        
        '''
        This script is mainly for reference purposes! It's not for forward pass.
        (Pdb) !torch.sum(att_einsum!=att)
        '''
        B, T, C = x.size()

        # att_einsum.shape = (c, BT, d)
        att_einsum = rearrange(x, "B T (c d) -> c (B T) d", B=B, T=T, c=self.c, d=self.d)

        # (c, BT, Hbs) = (c, BT, d) * (c, d, Hbs)
        att_einsum = self.bilinear_btt_wk(att_einsum)

        # att_einsum.shape = (Hb, cs, BT)
        att_einsum = rearrange(att_einsum, "c (B T) (H b s) -> (H b) (c s) (B T)", B=B, T=T, H=self.n_head, s=self.s, b=self.b, c=self.c)

        # (Hb, a, BT) = (Hb, a, cs) * (Hb, cs, BT)
        att_einsum = self.bilinear_btt_wq(att_einsum)

        # att_einsum.shape (B, HT, ab)
        att_einsum = rearrange(att_einsum, "(H b) a (B T) -> B (H T) (a b)", B=B, T=T, H=self.n_head, a=self.a, b=self.b)

        # att_einsum.shape = (B, T, HT) | Final Contraction
        att_einsum = torch.bmm(x, self.bilinear_btt_ln_PLPRXT(att_einsum).transpose(-2, -1))

        if self.bilinear_btt_muP_attn_logits_scaling:
            att_einsum = att_einsum * (1.0 / self.d_model) # muP attn logits scaling
        else:
            att_einsum = att_einsum * (1.0 / math.sqrt(self.d_model)) # SP attn logits scaling

        # att_einsum.shape = (B, H, T, T)
        att_einsum = rearrange(att_einsum, "B T (H N) -> B H T N", B=B, T=T, H=self.n_head, N=T)

        return att_einsum

    def forward(self, x, verify=False):
        if self.bilinearBTT_use_extra_LN_on_X:
            x = self.bilinear_btt_ln_X(x)

        B, T, C = x.size()

        # compute v_projection
        v = self.c_attn_v(x)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        ############################################ Bilinear BTT ############################################

        # att.shape = (c, BT, d)
        att = x.reshape(B*T, self.c, self.d).permute(1, 0, 2)

        # (c, BT, Hbs) = (c, BT, d) * (c, d, Hbs)
        att = self.bilinear_btt_wk(att)

        # att.shape = (Hb, cs, BT)
        att = att.reshape(self.c, B*T, self.n_head*self.b, self.s).permute(0, 3, 1, 2).reshape(self.c*self.s, B*T, self.n_head*self.b).permute(2, 0, 1)

        # (Hb, a, BT) = (Hb, a, cs) * (Hb, cs, BT)
        att = self.bilinear_btt_wq(att)

        # att.shape (B, HT, ab)
        att = att.reshape(self.n_head, self.b, self.a, B, T).permute(3, 0, 4, 2, 1).reshape(B, self.n_head*T, self.a*self.b)

        # att.shape = (B, T, HT) = (B, T, ab) * (B, ab, HT)
        att = torch.bmm(x, self.bilinear_btt_ln_PLPRXT(att).transpose(-2, -1))

        if self.bilinear_btt_muP_attn_logits_scaling:
            att = att * (1.0 / self.d_model) # muP attn logits scaling
        else:
            att = att * (1.0 / math.sqrt(self.d_model)) # SP attn logits scaling
        
        # att.shape = (B, H, T, T)
        att = att.reshape(B, T, self.n_head, T).permute(0, 2, 1, 3)

        # if verify:
        #     att_einsum = self.bilinear_btt_einsum(x)
        #     assert torch.sum(att!=att_einsum)==0
        #     print("pass assertion")
        

        ############################################ Bilinear BTT ############################################

        if self.link_function == "softmax":
            if self.causal: 
                att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
        elif self.link_function == "identity":
            if self.causal: 
                att = att.masked_fill(self.bias[:, :, :T, :T] == 0, 0)
        else:
            raise ValueError 
        
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class AttentionMLR(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_head == 0
        
        self.debug = config.debug
        self.mlr_divide_by_num_levels = config.mlr_divide_by_num_levels
        self.mlr_block_divide_by_num_levels = config.mlr_block_divide_by_num_levels

        # MLR rank distribution
        self.ranks = list(map(int, config.mlr_rank_list.split("|")))
        self.cumsum_ranks = torch.cumsum(torch.tensor([0] + self.ranks),dim=0).tolist()
        self.mlr_block_size_list = config.mlr_block_size_list
        self.block_size = config.block_size

        if self.mlr_block_size_list == "default":
            self.mlr_block_size_list = [self.block_size//(2**idx) for idx in range(len(self.ranks))]
        else:
            self.mlr_block_size_list = list(map(int, config.mlr_block_size_list.split("|")))

        assert len(self.ranks)==len(self.mlr_block_size_list)
        assert sum(self.ranks)==config.d_head

        for curr_block_size, curr_rank in zip(self.mlr_block_size_list, self.ranks):
            assert self.block_size % curr_block_size == 0
            assert curr_block_size >= curr_rank

        assert sum(self.ranks) == config.d_head

        self.split_qkv = config.split_qkv
        self.do_qk_ln = config.do_qk_ln
        self.link_function = config.link_function

        if self.do_qk_ln:
            self.ln_q = LayerNorm(config.d_head, bias=config.bias)
            self.ln_k = LayerNorm(config.d_head, bias=config.bias)

        if self.split_qkv:
            self.c_attn_q = nn.Linear(config.d_model, config.d_model, bias=config.bias)
            self.c_attn_k = nn.Linear(config.d_model, config.d_model, bias=config.bias)
            self.c_attn_v = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        else:
            self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)

        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.d_model = config.d_model
        self.dropout = config.dropout

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

        if self.mlr_block_divide_by_num_levels:
            mlr_block_mask = self.precompute_MLR_block_wise_division_mask()
            self.register_buffer(
                "mlr_block_mask",
                mlr_block_mask,
            )

    def precompute_MLR_block_wise_division_mask(self):
        mlr_block_mask = torch.zeros(1, 1, self.block_size, self.block_size)
        
        for i, curr_block_size in enumerate(self.mlr_block_size_list):
            num_blocks = int(self.block_size/curr_block_size)

            for j in range(num_blocks):
                mlr_block_mask[:, :, curr_block_size*j:curr_block_size*(j+1), curr_block_size*j:curr_block_size*(j+1)] += 1

        return 1.0 / mlr_block_mask
    
    def forward(self, x):
        B, T, C = x.shape

        # TODO: didn't implement inference for MLR Attention
        assert T==self.block_size

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        if self.split_qkv:
            q = self.c_attn_q(x)
            k = self.c_attn_k(x)
            v = self.c_attn_v(x)
        else:
            q, k, v  = self.c_attn(x).split(self.d_model, dim=2)
        
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # QK-layer normalization (per-head)
        if self.do_qk_ln:
            q = self.ln_q(q)
            k = self.ln_k(k)

        # split d_head based on "a|b|c|d"
        q_tuples = torch.split(q, self.ranks, dim=-1)
        k_tuples = torch.split(k, self.ranks, dim=-1)

        # MLR Matrix
        list_of_att_i = []

        for i, curr_block_size in enumerate(self.mlr_block_size_list):
            num_blocks = int(self.block_size/curr_block_size)
            att_i = torch.zeros(B, self.n_head, self.block_size, self.block_size, device=q.device, dtype=q.dtype)

            # split block_size based on curr_block_size
            curr_q = torch.stack(torch.split(q_tuples[i], curr_block_size, dim=-2))
            curr_k = torch.stack(torch.split(k_tuples[i], curr_block_size, dim=-2))

            curr_qk_product = torch.matmul(curr_q, curr_k.transpose(-2, -1)) * (1.0 / self.ranks[i])

            for j in range(num_blocks):
                att_i[:, :, curr_block_size*j:curr_block_size*(j+1), curr_block_size*j:curr_block_size*(j+1)] = curr_qk_product[j]

            list_of_att_i.append(att_i)

        if self.mlr_divide_by_num_levels:
            att = sum(list_of_att_i) * (1 / len(list_of_att_i))
        else:
            att = sum(list_of_att_i)
        
        if self.mlr_block_divide_by_num_levels:
            att = att * self.mlr_block_mask

        if self.link_function == "softmax":
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
        elif self.link_function == "identity":
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, 0)
        else:
            raise ValueError
        
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y



class BilinearMLRProjection(nn.Module):
    '''Assume m_lk = n_lk'''
    def __init__(self, shape):
        super().__init__()
        # self.weight.shape = (p_l, m_lk, H * r_l)
        self.weight = nn.Parameter(torch.randn(shape))

    def forward(self, x):
        '''x.shape = (p_l, B * T, m_lk)'''
        # TODO: check on this RMS normalization
        if True:
            rms_curr_w_l = torch.sqrt(torch.mean(self.weight**2.) + 1e-8)
            d_in = self.weight.size(-2)
            d_out = self.weight.size(-1)
            max_rms_rms_curr_w_l = (min(d_in, d_out) * d_out / (d_out * d_in * d_in))**0.5
            curr_wq_l_normed = self.weight / max(1, rms_curr_w_l / max_rms_rms_curr_w_l)

        # output.shape = (p_l, B * T, H * r_l)
        return torch.bmm(x, curr_wq_l_normed)

class BilinearMLRAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_head == 0
        
        self.debug = config.debug
        self.mlr_divide_by_num_levels = config.mlr_divide_by_num_levels
        self.mlr_block_divide_by_num_levels = config.mlr_block_divide_by_num_levels
        self.bilinear_mlr_muP_attn_logits_scaling = config.bilinear_mlr_muP_attn_logits_scaling

        # MLR rank distribution
        self.ranks = list(map(int, config.mlr_rank_list.split("|")))
        self.cumsum_ranks = torch.cumsum(torch.tensor([0] + self.ranks),dim=0).tolist()
        self.list_of_num_blocks = [2**idx for idx in range(len(self.ranks))]
        assert sum(self.ranks) == config.d_head

        self.split_qkv = config.split_qkv
        self.do_qk_ln = config.do_qk_ln
        self.link_function = config.link_function
        
        # (p_l, m_lk, H * r_l)
        self.bilinear_mlr_wq = nn.ModuleList([BilinearMLRProjection((p_l, config.d_model // p_l, config.n_head * r_l)) for r_l, p_l in zip(self.ranks, self.list_of_num_blocks)])
        
        # (p_l, n_lk, H * r_l)
        self.bilinear_mlr_wk = nn.ModuleList([BilinearMLRProjection((p_l, config.d_model // p_l, config.n_head * r_l)) for r_l, p_l in zip(self.ranks, self.list_of_num_blocks)])

        if self.do_qk_ln:
            self.bilinear_mlr_ln_wq = nn.ModuleList([LayerNorm(r_l*p_l, bias=config.bias) for r_l, p_l in zip(self.ranks, self.list_of_num_blocks)])
            self.bilinear_mlr_ln_wk = nn.ModuleList([LayerNorm(r_l*p_l, bias=config.bias) for r_l, p_l in zip(self.ranks, self.list_of_num_blocks)])

        assert self.split_qkv==True
        self.c_attn_v = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.d_model = config.d_model
        self.block_size = config.block_size
        self.dropout = config.dropout

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

        if self.mlr_block_divide_by_num_levels:
            raise NotImplementedError

            # mlr_block_mask = self.precompute_MLR_block_wise_division_mask()
            # self.register_buffer(
            #     "mlr_block_mask",
            #     mlr_block_mask,
            # )

    # def precompute_MLR_block_wise_division_mask(self):
    #     mlr_block_mask = torch.zeros(1, 1, self.block_size, self.block_size)
        
    #     for i, num_blocks in enumerate(self.list_of_num_blocks):
    #         curr_block_size = int(self.block_size/num_blocks)

    #         for j in range(num_blocks):
    #             mlr_block_mask[:, :, curr_block_size*j:curr_block_size*(j+1), curr_block_size*j:curr_block_size*(j+1)] += 1

    #     return 1.0 / mlr_block_mask
    
    def forward(self, x):
        B, T, C = x.shape

        # compute v_projection
        v = self.c_attn_v(x)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # initialize storage
        att = torch.zeros(B, self.n_head, T, T, device=v.device, dtype=v.dtype)

        # for loop over number of levels in MLR
        for curr_l, (r_l, p_l) in enumerate(zip(self.ranks, self.list_of_num_blocks)):            
            # curr_x.shape -> (p_l, BT, m_lk)
            curr_x = torch.stack(torch.split(x, C // p_l, dim=-1),dim=0).reshape(p_l, B*T, C//p_l)

            # curr_q.shape -> (p_l, BT, H r_l)
            curr_q = self.bilinear_mlr_wq[curr_l](curr_x)

            # curr_k.shape -> (p_l, BT, H r_l)
            curr_k = self.bilinear_mlr_wk[curr_l](curr_x)

            # curr_q.shape: "p (B T) (H r) -> B H T (r p)"
            curr_q = curr_q.reshape(p_l, B, T, self.n_head, r_l).permute(1, 3, 2, 4, 0).reshape(B, self.n_head, T, r_l * p_l)

            # curr_k.shape: "p (B T) (H r) -> B H T (r p)"
            curr_k = curr_k.reshape(p_l, B, T, self.n_head, r_l).permute(1, 3, 2, 4, 0).reshape(B, self.n_head, T, r_l * p_l)

            # QK LayerNorm
            if self.do_qk_ln:
                curr_ln_q = self.bilinear_mlr_ln_wq[curr_l]
                curr_ln_k = self.bilinear_mlr_ln_wk[curr_l]

                curr_q = curr_ln_q(curr_q)
                curr_k = curr_ln_k(curr_k)

            attn_logit_scaling = (1 / math.sqrt(r_l*p_l))

            # attention logit scaling
            if self.bilinear_mlr_muP_attn_logits_scaling:
                attn_logit_scaling = attn_logit_scaling * (8 / math.sqrt(r_l*p_l))
            
            # curr_qk_product.shape = (B, H, T T)
            curr_qk_product = torch.matmul(curr_q, curr_k.transpose(-2, -1)) * attn_logit_scaling

            # perform summation
            att = att + curr_qk_product

        if self.mlr_divide_by_num_levels:
            att = att * (1 / len(self.ranks))
        else:
            pass
                
        if self.mlr_block_divide_by_num_levels:
            raise NotImplementedError
        
            # att = att * self.mlr_block_mask

        if self.link_function == "softmax":
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
        elif self.link_function == "identity":
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, 0)
        else:
            raise ValueError
                
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y



class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.d_model, 4 * config.d_model, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.d_model, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config, i_layer):
        super().__init__()
        self.token_mixing_struct = config.token_mixing_struct
        self.ln_1 = LayerNorm(config.d_model, bias=config.bias)
        
        if self.token_mixing_struct == "low_rank":
            self.attn = CausalSelfAttention(config)
        elif self.token_mixing_struct == "low_rank_with_sliding_window":
            self.attn = CausalSelfSlidingWindowAttention(config)
        elif self.token_mixing_struct == "low_rank_with_gswa":
            self.attn = CausalSelfGlobalSlidingWindowAttention(config)
        elif self.token_mixing_struct == "low_rank_with_sequential_gswa":
            assert config.n_layer == len(config.sequential_GSWA_layer_list.split("|"))

            if config.sequential_GSWA_layer_list.split("|")[i_layer] == "MHA":
                self.attn = CausalSelfAttention(config)
            elif config.sequential_GSWA_layer_list.split("|")[i_layer] == "SWA":
                self.attn = CausalSelfSlidingWindowAttention(config)
            else:
                raise ValueError
        elif self.token_mixing_struct == "multi_level_low_rank":
            self.attn = AttentionMLR(config)
        elif self.token_mixing_struct == "bilinear_MLR":
            self.attn = BilinearMLRAttention(config)
        elif self.token_mixing_struct == "bilinear_BTT":
            self.attn = BilinearBTTAttention(config)
        else:
            raise ValueError
        
        self.ln_2 = LayerNorm(config.d_model, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
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

class GPT(nn.Module):

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

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.d_model),
            wpe = nn.Embedding(config.block_size, config.d_model),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config, i_layer) for i_layer in range(config.n_layer)]),
            ln_f = LayerNorm(config.d_model, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, d_model)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, d_model)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.d_model//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
