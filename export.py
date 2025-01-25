import os
import torch
import struct
import numpy as np


def serialize_fp32(file, tensor):
    """Writes one fp32 tensor to file that is open in wb mode"""
    d = tensor.detach().cpu().view(-1).to(torch.float32).numpy()
    b = struct.pack(f'{len(d)}f', *d)
    file.write(b)


def serialize_int8(file, tensor):
    """Writes one int8 tensor to file that is open in wb mode"""
    d = tensor.detach().cpu().view(-1).numpy().astype(np.int8)
    b = struct.pack(f'{len(d)}b', *d)
    file.write(b)


def version1_export(model, filepath):
    """Exports the model in full float32 format"""
    out_file = open(filepath, "wb")
    # Write header
    out_file.write(struct.pack('I', 0x616b3432))  # magic number
    out_file.write(struct.pack('i', 1))  # version
    # Write model parameters
    p = model.params
    out_file.write(struct.pack('iiiiiii', p.dim, p.n_layers,
                   p.n_heads, p.n_kv_heads, p.vocab_size, p.max_seq_len, 0))
    # Write weights
    weights = [
        model.tok_embeddings.weight,
        *[layer.attention.wq.weight for layer in model.layers],
        *[layer.attention.wk.weight for layer in model.layers],
        *[layer.attention.wv.weight for layer in model.layers],
        *[layer.attention.wo.weight for layer in model.layers],
        *[layer.feed_forward.w1.weight for layer in model.layers],
        *[layer.feed_forward.w2.weight for layer in model.layers],
        *[layer.feed_forward.w3.weight for layer in model.layers],
        model.norm.weight,
    ]
    for w in weights:
        serialize_fp32(out_file, w)
    out_file.close()
    print(f"Exported model to {filepath}")


def model_export(model, filepath, version):
    if version == 1:
        version1_export(model, filepath)
    else:
        raise ValueError(f"Unknown version {version}")
