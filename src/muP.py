import torch
import torch.nn as nn
from torch.optim import AdamW

import inspect

def compute_muP_lr(name, fan_out, fan_in, base_width_factor, lr, opt_name):
    if opt_name == "AdamW":
        base_lr = lr * base_width_factor / fan_in
        print(f"### {name} base_lr = lr / fan_in = {lr} * {base_width_factor} / {fan_in} = {base_lr} ###\n")
    else:
        raise ValueError
    
    return base_lr

def muPify(
        model, 
        device, device_type,
        # optimization arguments
        opt_name, lr, weight_decay, beta1, beta2, init_from='scratch',
    ):
    '''
    Maximum Update Parametrization (muP) is a technique proposed in Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer 
    (https://arxiv.org/abs/2203.03466) for optimal parametrizations of initialization and learning rates. 

    Table 3 in https://arxiv.org/abs/2203.03466:

    |                | Input weights & all biases | Output weights      | Hidden weights  |
    |----------------|----------------------------|---------------------|-----------------|
    | Init. Var.     | 1 / fan_in                 | 1 / fan_in^2        | 1 / fan_in      |
    | SGD LR         | fan_out                    | 1 / fan_in          | 1               |
    | Adam LR        | 1                          | 1 / fan_in          | 1 / fan_in      |
    '''

    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}

    param_groups = []
    peak_lrs = {}  # store base learning rates by param name

    for name, param in param_dict.items():
        ########
        # ViT
        ########
        if "cls_token" in name:
            fan_out, fan_in = None, 1
            base_width_factor = 1
            c_read_in = 0.02 # tunable constant
            
            # muP initialization
            if init_from == 'scratch':
                nn.init.normal_(param, mean=0.0, std=c_read_in * 1.0 / (fan_in ** 0.5))
                print(f"### {name} initialized with std = {c_read_in} * 1.0 / ({fan_in} ** 0.5) = {c_read_in * 1.0 / (fan_in ** 0.5)} ###")
            elif init_from == 'resume':
                print(f"skip muP re-initialization for resumed training runs")
            else:
                raise ValueError
        # layer norms
        elif any(module_name in name for module_name in ['to_patch_embedding.1.weight', 'to_patch_embedding.3.weight', 'norm.weight', 'fnn2layer.0.weight']):
            fan_out, fan_in = None, 1 # set fan_in to 1 for LayerNorm learning rate scaling
            base_width_factor = 1

            # muP initialization
            if init_from == 'scratch':
                nn.init.ones_(param)
                print(f"### {name} initialized with ones ###")
            elif init_from == 'resume':
                print(f"skip muP re-initialization for resumed training runs")
            else:
                raise ValueError
        elif name == "to_patch_embedding.2.weight":
            # input embedding. should not scale with width of the network
            # But muP paper appendix B.1, "Input Word Embeddings" says to use 1/fan_in for image inputs
            fan_out, fan_in = None, param.shape[1]
            base_width_factor = 768
            
            # muP initialization
            if init_from == 'scratch':
                nn.init.normal_(param, mean=0.0, std=c_read_in * 1.0 / (fan_in ** 0.5))
                print(f"### {name} initialized with std = {c_read_in} * 1.0 / ({fan_in} ** 0.5) = {c_read_in * 1.0 / (fan_in ** 0.5)} ###")
            elif init_from == 'resume':
                print(f"skip muP re-initialization for resumed training runs")
            else:
                raise ValueError
        elif name == "mlp_head.weight":
            fan_out, fan_in = param.shape
            base_width_factor = 768
            
            # muP initialization
            if init_from == 'scratch':
                nn.init.zeros_(param)
                print(f"### {name} initialized with zeros ###")
            elif init_from == 'resume':
                print(f"skip muP re-initialization for resumed training runs")
            else:
                raise ValueError
        ########
        # Transformer
        ########
        elif "wte" in name:
            fan_out, fan_in = None, 1 # one-hot lookup
            base_width_factor = 1
            c_wte = 0.02 # tunable constant
            
            # muP initialization
            if init_from == 'scratch':
                nn.init.normal_(param, mean=0.0, std=c_wte * 1.0 / (fan_in ** 0.5))
                print(f"### {name} initialized with std = {c_wte} * 1.0 / ({fan_in} ** 0.5) = {c_wte * 1.0 / (fan_in ** 0.5)} ###")
            elif init_from == 'resume':
                print(f"skip muP re-initialization for resumed training runs")
            else:
                raise ValueError
        elif ("wpe" in name) or ("pos_embedding" in name):
            fan_out, fan_in = None, 1 # one-hot lookup
            base_width_factor = 1
            c_wpe = 0.02 # tunable constant

            # muP initialization
            if init_from == 'scratch':
                nn.init.normal_(param, mean=0.0, std=c_wpe * 1.0 / (fan_in ** 0.5))
                print(f"### {name} initialized with std = {c_wpe} * 1.0 / ({fan_in} ** 0.5) = {c_wpe * 1.0 / (fan_in ** 0.5)} ###")
            elif init_from == 'resume':
                print(f"skip muP re-initialization for resumed training runs")
            else:
                raise ValueError
        elif any(module_name in name for module_name in ['ln_1.weight', 'ln_2.weight', 'ln_f.weight', 'ln_q.weight', 'ln_k.weight', 'bilinear_mlr_ln_wq', 'bilinear_mlr_ln_wk', 'bilinear_btt_ln_PLPRXT.weight', 'bilinear_btt_ln_X.weight']):
            assert (not hasattr(model, "config")) or model.config.bias==False, "bilinear_mlr_ln_wq & bilinear_mlr_ln_wk assume bias=False!!!"

            fan_out, fan_in = None, 1 # set fan_in to 1 for LayerNorm learning rate scaling
            base_width_factor = 1

            # muP initialization
            if init_from == 'scratch':
                nn.init.ones_(param)
                print(f"### {name} initialized with ones ###")
            elif init_from == 'resume':
                print(f"skip muP re-initialization for resumed training runs")
            else:
                raise ValueError
        elif any(module_name in name for module_name in ['attn.c_attn_q.weight', 'attn.c_attn_k.weight', 'attn.c_attn_v.weight', '.0.c_attn_v.weight', 'mlp.c_fc.weight', 'to_k.weight', 'to_v.weight', 'fnn2layer.1.weight']):
            fan_out, fan_in = param.shape
            base_width_factor = 768

            # muP initialization
            if init_from == 'scratch':
                nn.init.normal_(param, mean=0.0, std=1.0 / (fan_in ** 0.5))
                print(f"### {name} initialized with std = 1.0 / ({fan_in} ** 0.5)={1.0 / (fan_in ** 0.5)} ###")
            elif init_from == 'resume':
                print(f"skip muP re-initialization for resumed training runs")
            else:
                raise ValueError
        elif any(module_name in name for module_name in ['attn.bilinear_mlr_wq', 'attn.bilinear_mlr_wk']):
            # (p_l, m_lk, H * r_l)
            p_l, m_lk, Hr_l = param.shape # assuming m_lk = n_lk = d_model / p_l

            fan_out = Hr_l
            fan_in = m_lk
            base_width_factor = 768

            # muP initialization
            if init_from == 'scratch':
                nn.init.normal_(param, mean=0.0, std=1.0 / (fan_in ** 0.5))
                print(f"### {name} initialized with std = 1.0 / ({fan_in} ** 0.5)={1.0 / (fan_in ** 0.5)} ###")
            elif init_from == 'resume':
                print(f"skip muP re-initialization for resumed training runs")
            else:
                raise ValueError
        elif name.endswith("attn.bilinear_btt_wk.weight") or name.endswith(".0.bilinear_btt_wk.weight"):
            # (c, d, Hbs)
            _, fan_in, fan_out = param.shape
            base_width_factor = 768

            # muP initialization
            if init_from == 'scratch':
                nn.init.normal_(param, mean=0.0, std=1.0 / (fan_in ** 0.5))
                print(f"### {name} initialized with std = 1.0 / ({fan_in} ** 0.5)={1.0 / (fan_in ** 0.5)} ###")
            elif init_from == 'resume':
                print(f"skip muP re-initialization for resumed training runs")
            else:
                raise ValueError
        elif name.endswith("attn.bilinear_btt_wq.weight") or name.endswith(".0.bilinear_btt_wq.weight"):
            # (bH, a, cs)
            _, fan_out, fan_in = param.shape
            base_width_factor = 768

            # muP initialization
            if init_from == 'scratch':
                nn.init.normal_(param, mean=0.0, std=1.0 / (fan_in ** 0.5))
                print(f"### {name} initialized with std = 1.0 / ({fan_in} ** 0.5)={1.0 / (fan_in ** 0.5)} ###")
            elif init_from == 'resume':
                print(f"skip muP re-initialization for resumed training runs")
            else:
                raise ValueError
        elif any(module_name in name for module_name in ['attn.c_proj.weight', '.0.c_proj.weight', 'to_q.weight', 'to_out.0.weight', 'mlp.c_proj.weight', 'fnn2layer.4.weight']):
            fan_out, fan_in = param.shape
            base_width_factor = 768

            # muP initialization
            if init_from == 'scratch':
                nn.init.zeros_(param)
                print(f"### {name} initialized with zeros ###")
            elif init_from == 'resume':
                print(f"skip muP re-initialization for resumed training runs")
            else:
                raise ValueError
        elif "lm_head" in name:
            fan_out, fan_in = param.shape
            base_width_factor = 768

            # muP initialization
            if init_from == 'scratch':
                nn.init.zeros_(param)
                print(f"### {name} initialized with zeros ###")
            elif init_from == 'resume':
                print(f"skip muP re-initialization for resumed training runs")
            else:
                raise ValueError
        ########
        # ICL Regression
        ########
        elif "_read_in.weight" in name: # ICL Regression
            fan_out, fan_in = None, 1
            base_width_factor = 1
            c_read_in = 0.02 # tunable constant
            
            # muP initialization
            if init_from == 'scratch':
                nn.init.normal_(param, mean=0.0, std=c_read_in * 1.0 / (fan_in ** 0.5))
                print(f"### {name} initialized with std = {c_read_in} * 1.0 / ({fan_in} ** 0.5) = {c_read_in * 1.0 / (fan_in ** 0.5)} ###")
            elif init_from == 'resume':
                print(f"skip muP re-initialization for resumed training runs")
            else:
                raise ValueError
        elif "_read_out.weight" in name: # ICL Regression
            fan_out, fan_in = param.shape
            base_width_factor = 768

            # muP initialization
            if init_from == 'scratch':
                nn.init.zeros_(param)
                print(f"### {name} initialized with zeros ###")
            elif init_from == 'resume':
                print(f"skip muP re-initialization for resumed training runs")
            else:
                raise ValueError
        ########
        # all bias weights are the same
        ########
        elif name.endswith(".bias"):
            fan_out, fan_in = None, 1
            base_width_factor = 1

            # muP initialization
            if init_from == 'scratch':
                nn.init.zeros_(param)
                print(f"### {name} initialized with zeros ###")
            elif init_from == 'resume':
                print(f"skip muP re-initialization for resumed training runs")
            else:
                raise ValueError
        else:
            raise ValueError(f"{name} not compatible")

        # muP learning rate
        base_lr = compute_muP_lr(name, fan_out, fan_in, base_width_factor, lr, opt_name)

        group = {
            'params': param, 
            'weight_decay': weight_decay if param.dim() >= 2 else 0,
            'lr': base_lr, 
            'name': name}
        peak_lrs[name] = base_lr
        param_groups.append(group)

    # send model to device
    model.to(device)

    # Log parameter counts and learning rates
    decay_params = [g for g in param_groups if g['weight_decay'] > 0]
    nodecay_params = [g for g in param_groups if g['weight_decay'] == 0]
    
    num_decay_params = sum(p.numel() for g in decay_params for p in g['params'])
    num_nodecay_params = sum(p.numel() for g in nodecay_params for p in g['params'])

    print(f"num_decay_params={num_decay_params}")
    print(f"num_nodecay_params={num_nodecay_params}")

    if opt_name == "AdamW":
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()

        optimizer = AdamW(param_groups, betas=(beta1, beta2), eps=1e-20, **extra_args)
        print(f"\nusing fused AdamW: {use_fused}")

        for group in param_groups:
            print(f"{group['name']} | lr ={group['lr']}")
        
    return model, optimizer

