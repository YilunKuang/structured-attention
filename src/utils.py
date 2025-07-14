import os
import random
import string
import datetime

import torch
from torchinfo import summary
from fvcore.nn import FlopCountAnalysis

def generate_out_dir(base_path):
    # Get current timestamp with precision to seconds
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Generate a random string of 6 characters
    random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
    
    # Combine components to form the directory
    out_dir = os.path.join(base_path, f"checkpoint_{timestamp}_{random_str}")
    
    return out_dir

def get_model_summary_and_flops(model, fake_input):
    print('Model:')
    stats = summary(model, input_data=fake_input)
    cola_params = stats.trainable_params
    print(f'Params: {cola_params / 1e6:.2f}M')
    cola_flops = FlopCountAnalysis(model, fake_input).set_op_handle(**custom_ops).total()
    print(f'FLOPs: {cola_flops / 1e6:.2f}M')
    # FlopCountAnalysis(model, fake_input).set_op_handle(**custom_ops).by_module()
    # FlopCountAnalysis(model, fake_input).set_op_handle(**custom_ops).by_module_and_operator()
    neurons = 0  # count_neurons(model, fake_input)
    print(f"Neurons: {neurons}")
    print('=' * 90)
    info = {'cola_params': cola_params, 'cola_flops': cola_flops, "neurons": neurons}

    return info

def btt_flop_count(inputs, outputs):
    if len(inputs) == 3:
        x, W1, W2 = inputs
        batch_size = get_shape(x)[0]
        flops = get_numel(W1) + get_numel(W2)
    elif len(inputs) == 5:
        x, W1, W2, gate, num_active = inputs
        num_experts = get_shape(gate)[1]
        num_active_experts = get_shape(num_active)[0]
        batch_size = get_shape(x)[0]
        flops = get_numel(W1) + get_numel(W2)
        flops = flops * (num_active_experts / num_experts)
    else:
        raise ValueError(f'Unexpected number of inputs: {len(inputs)}')
    return batch_size * flops


def scaled_dot_product_attention_flop_count(inputs, outputs):
    output_shape = get_shape(outputs[0])  # ([batch dims], seq_len, head_dim)
    B = prod(output_shape[:-2])  # batch suze
    L, D = output_shape[-2], output_shape[-1]  # (seq_len, head_dim)
    qk_flops = B * (L * D * L)  # (L, D) @ (D, L)
    v_flops = B * (L * L * D)  # (L, L) @ (L, D)
    return qk_flops + v_flops

def custom_einsum_flop_count(inputs, outputs):
    """
    Count flops for the einsum operation.
    """
    # Inputs of einsum should be a list of length 2+.
    # Inputs[0] stores the equation used for einsum.
    # Inputs[1] stores the list of input shapes.
    # Inputs[2] optionally stores the optimized path of contraction.
    assert len(inputs) >= 2, len(inputs)
    equation = inputs[0].toIValue()
    # Get rid of white space in the equation string.
    equation = equation.replace(" ", "")
    input_shapes_jit = inputs[1].node().inputs()
    input_shapes = [get_shape(v) for v in input_shapes_jit]

    # Re-map equation so that same equation with different alphabet
    # representations will look the same.
    letter_order = OrderedDict((k, 0) for k in equation if k.isalpha()).keys()
    mapping = {ord(x): 97 + i for i, x in enumerate(letter_order)}
    equation = equation.translate(mapping)

    if equation == "abc,abd->acd":
        n, c, t = input_shapes[0]
        p = input_shapes[-1][-1]
        flop = n * c * t * p
        return flop

    elif equation == "abc,adc->adb":
        n, t, g = input_shapes[0]
        c = input_shapes[-1][1]
        flop = n * t * g * c
        return flop
    else:
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = opt_einsum.contract_path(equation, *np_arrs, optimize='optimal')[1]
        return int(optim.opt_cost) / 2

def compute_model_FLOPs(
        model,
        config,
        vocab_size,
        block_size,
        device,
    ):
    fake_input = torch.randint(low=0, high=vocab_size, size=(1, block_size)).to(device)
    info = get_model_summary_and_flops(model, (fake_input, fake_input))
    emb_params = sum([p.numel() for name, p in model.named_parameters() if 'wte' in name or 'wpe' in name])
    head_params = sum([p.numel() for name, p in model.named_parameters() if 'lm_head' in name])
    info['emb_params'] = emb_params
    info['head_params'] = head_params
    info['non_emb_params'] = info['cola_params'] - emb_params - head_params  # i.e. non-embedding params
    param_str = f'P: {info["cola_params"]/1e6:.2f} M | E: {emb_params/1e6:.2f} M | H: {head_params/1e6:.2f} M |'
    param_str += f' Non-embd: {info["non_emb_params"]/1e6:.2f} M'
    flops = info['cola_flops']
    flops_per_token = flops / block_size
    non_emb_flops = flops - head_params * block_size  # exclude emb and unemb
    non_emb_flops_per_token = non_emb_flops / block_size
    info['non_emb_flops'] = non_emb_flops
    print(param_str)
    print(f'Non-emb FLOPs: {non_emb_flops // 1e6} M')
    print(f"info={info}")
    config.update(info)
    return flops_per_token, non_emb_flops_per_token

def compute_model_FLOPs_for_ICLRegression(
        model,
        curr_n_positions,
        curr_n_dims,
        device,
    ):
    fake_input = torch.randn(size=(1, curr_n_positions, curr_n_dims)).to(device)
    fake_target = torch.randn(size=(1, curr_n_positions, 1)).to(device)
    info = get_model_summary_and_flops(model, (fake_input, fake_target))
    emb_params = sum([p.numel() for name, p in model.named_parameters() if 'wte' in name or 'wpe' in name])
    head_params = sum([p.numel() for name, p in model.named_parameters() if 'lm_head' in name])
    info['emb_params'] = emb_params
    info['head_params'] = head_params
    info['non_emb_params'] = info['cola_params'] - emb_params - head_params  # i.e. non-embedding params
    param_str = f'P: {info["cola_params"]/1e6:.2f} M | E: {emb_params/1e6:.2f} M | H: {head_params/1e6:.2f} M |'
    param_str += f' Non-embd: {info["non_emb_params"]/1e6:.2f} M'
    flops = info['cola_flops']
    flops_per_token = flops / (curr_n_positions*2)
    non_emb_flops = flops - head_params * (curr_n_positions*2)  # exclude emb and unemb
    non_emb_flops_per_token = non_emb_flops / (curr_n_positions*2)
    info['non_emb_flops'] = non_emb_flops
    print(param_str)
    print(f'Non-emb FLOPs: {non_emb_flops // 1e6} M')
    return flops_per_token, non_emb_flops_per_token

custom_ops = {
    'prim::PythonOp.BlockdiagButterflyMultiply': btt_flop_count,
    'prim::PythonOp.BlockTeTr': btt_flop_count,
    'prim::PythonOp.RieBTTmvm': btt_flop_count,
    'prim::PythonOp.BTTGen': btt_flop_count,
    'prim::PythonOp.BTTmvm': btt_flop_count,
    'aten::scaled_dot_product_attention': scaled_dot_product_attention_flop_count,
    'aten::einsum': custom_einsum_flop_count,
}