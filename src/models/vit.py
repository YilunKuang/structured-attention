# adapted from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from ..model import BilinearBTTAttention

class CappedList():
    # used for caching activations for logging
    def __init__(self, max_len=1):
        self.max_len = max_len
        self.buffer = []

    def append(self, x):
        if len(self.buffer) < self.max_len:
            self.buffer.append(x.cpu())


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0., fixup=False, use_bias=True):
        super().__init__()
        self.fnn2layer = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden_dim, bias=use_bias), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden_dim, dim, bias=use_bias), nn.Dropout(dropout))

        # a scaler multiplier
        self.out_scalar = nn.Parameter(torch.ones(1)) if fixup else None


    def forward(self, x):
        out = self.fnn2layer(x)
        if self.out_scalar is not None:
            out = out * self.out_scalar
        return out


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., fixup=False, attn_mult=1, use_bias=True, causal=False, qknorm=True):
        super().__init__()
        self.causal = causal
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = attn_mult * 8 / dim_head  # μP prescribs this scaling, 8 for backward compatibility at dim_head=64

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        # split into 3 separate linears so that CoLA replaces each individually
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.q_norm = nn.LayerNorm(dim_head) if qknorm else None
        self.k_norm = nn.LayerNorm(dim_head) if qknorm else None

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim, bias=use_bias),
                                    nn.Dropout(dropout)) if project_out else nn.Identity()

        # a scaler multiplier
        self.out_scalar = nn.Parameter(torch.ones(1)) if fixup else None

    def forward(self, x):
        x = self.norm(x)

        qkv = (self.to_q(x), self.to_k(x), self.to_v(x))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        if self.q_norm is not None: q = self.q_norm(q)
        if self.k_norm is not None: k = self.k_norm(k)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # b h n n

        # attention mask
        if self.causal:
            mask = torch.ones(dots.shape[-2], dots.shape[-1], device=dots.device).triu(1)
            dots = dots.masked_fill(mask == 1, float('-inf'))

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        if self.out_scalar is not None:
            out = out * self.out_scalar
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., fixup=False, attn_mult=1,
                 use_bias=True, causal=False, last_layernorm=True, alt_attn_config=None, bilinearBTT_use_extra_LN_on_X=1):
        super().__init__()
        self.norm = nn.LayerNorm(dim) if last_layernorm else nn.Identity()
        self.layers = nn.ModuleList([])
        self.fixup = fixup
        self.attn_mult = attn_mult
        for _ in range(depth):
            ffn = FeedForward(dim, mlp_dim, dropout=dropout, fixup=fixup, use_bias=use_bias)
            if ((alt_attn_config is None) or (alt_attn_config == "none")):
                attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout, fixup=fixup, attn_mult=attn_mult,
                                use_bias=use_bias, causal=causal, qknorm=True)  # TODO: could add config for qknorm
            elif alt_attn_config.startswith("BTT"):
                _, btt_tt_dim, btt_tt_rank, bilinear_btt_muP_attn_logits_scaling = alt_attn_config.split(",")
                attn = BilinearBTTAttention(SimpleNamespace(
                    d_model=dim,
                    n_head=heads,
                    d_qk_head=dim_head,
                    causal=causal,
                    dropout=dropout,
                    bias=use_bias,
                    btt_tt_dim=int(btt_tt_dim),
                    btt_tt_rank=int(btt_tt_rank),
                    bilinear_btt_muP_attn_logits_scaling=eval(bilinear_btt_muP_attn_logits_scaling),
                    split_qkv=True,
                    manual_disable_flash_att=True,
                    do_qk_ln=True,
                    link_function="softmax",
                    bilinearBTT_use_extra_LN_on_X=bilinearBTT_use_extra_LN_on_X,
                ))
            else:
                raise NotImplementedError(f"{alt_attn_config} not an attn type")
            self.layers.append(nn.ModuleList([attn, ffn]))
        self.hs = [CappedList() for _ in range(depth + 2)]

    def forward(self, x):
        if not self.training:
            self.hs[0].append(x.detach())
        for i, (attn, ff) in enumerate(self.layers):
            x = attn(x) + x
            x = ff(x) + x
            if not self.training:
                self.hs[i + 1].append(x.detach())
        return self.norm(x)


class ViT(nn.Module):
    def __init__(self, dim_out, width, depth, ffn_expansion=4, heads=8, dim_head=None, image_size=32, patch_size=8, pool='cls',
                 in_channels=3, dropout=0., fixup=False, attn_mult=1, output_mult=1, emb_mult=1, use_bias=True, bilinearBTT_use_extra_LN_on_X=1, 
                 **kwargs):
        super().__init__()
        self.fixup = fixup
        self.emb_mult = emb_mult
        self.attn_mult = attn_mult
        self.output_mult = output_mult
        if (dim_head is None) and (kwargs.get("alt_attn_config", "")[:3] != "QK="):
            dim_head = width / heads
            assert int(dim_head) == dim_head, 'dimension of each head must be integer'
            dim_head = int(dim_head)
            print(f"Setting dim_head to {dim_head}")
        mlp_dim = ffn_expansion * width
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Invalid patch size'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = in_channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, width, bias=use_bias),
            nn.LayerNorm(width),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, width))
        self.pos_embedding.fan_in_dims = (1)  # constant fan in
        self.cls_token = nn.Parameter(torch.randn(1, 1, width))
        self.cls_token.fan_in_dims = (1)  # constant fan in
        self.dropout = nn.Dropout(dropout)

        self.transformer = Transformer(width, depth, heads, dim_head, mlp_dim, dropout, fixup, attn_mult,
                                    use_bias, alt_attn_config=kwargs.get("alt_attn_config", None), bilinearBTT_use_extra_LN_on_X=bilinearBTT_use_extra_LN_on_X)

        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Linear(width, dim_out, bias=use_bias)

        if not use_bias:
            # Freeze all emb and pos embeddings
            emb_params = [self.pos_embedding, self.cls_token] + list(self.to_patch_embedding.parameters())
            for p in emb_params:
                p.requires_grad = False

        # logs
        self.hs = [CappedList() for _ in range(depth + 2)]

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = x * self.emb_mult
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        y = self.mlp_head(x) * self.output_mult
        if not self.training:
            self.transformer.hs[-1].append(y.detach())
        return y

    def get_features(self):
        return self.transformer.hs

    def clear_features(self):
        self.transformer.hs = [CappedList() for _ in range(len(self.hs))]