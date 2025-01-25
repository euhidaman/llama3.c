import os
import struct
import torch
from model import ModelArgs, Transformer


def serialize_fp32(file, tensor):
    d = tensor.detach().cpu().view(-1).to(torch.float32).numpy()
    b = struct.pack(f'{len(d)}f', *d)
    file.write(b)


def serialize_int8(file, tensor):
    d = tensor.detach().cpu().view(-1).numpy().astype(np.int8)
    b = struct.pack(f'{len(d)}b', *d)
    file.write(b)


def quantize_q80(w, group_size):
    assert w.numel() % group_size == 0
    ori_shape = w.shape
    w = w.float()
    w = w.reshape(-1, group_size)
    wmax = torch.abs(w).max(dim=1).values
    scale = wmax / 127.0
    quant = w / scale[:, None]
    int8val = torch.round(quant).to(torch.int8)
    fp32val = (int8val.float() * scale[:, None]).view(-1)
    fp32valr = fp32val.reshape(-1, group_size)
    err = torch.abs(fp32valr - w).max(dim=1).values
    maxerr = err.max().item()
    return int8val, scale, maxerr


def version1_export(model, filepath):
    version = 1
    out_file = open(filepath, 'wb')
    out_file.write(struct.pack('I', 0x616b3432))  # Magic number
    out_file.write(struct.pack('i', version))
    p = model.params
    hidden_dim = model.layers[0].feed_forward.w3.weight.shape[0]
    n_kv_heads = p.n_kv_heads
    header = struct.pack('iiiiiii', p.dim, hidden_dim, p.n_layers,
                         p.n_heads, n_kv_heads, p.vocab_size, p.max_seq_len)
    out_file.write(header)
    shared_classifier = torch.equal(
        model.tok_embeddings.weight, model.output.weight)
    out_file.write(struct.pack('B', int(shared_classifier)))
    pad = 256 - out_file.tell()
    assert pad >= 0
    out_file.write(b'\0' * pad)

    weights = [
        *[layer.attention_norm.weight for layer in model.layers],
        *[layer.ffn_norm.weight for layer in model.layers],
        model.norm.weight,
        model.tok_embeddings.weight,
        *[layer.attention.wq.weight for layer in model.layers],
        *[layer.attention.wk.weight for layer in model.layers],
        *[layer.attention.wv.weight for layer in model.layers],
        *[layer.attention.wo.weight for layer in model.layers],
        *[layer.feed_forward.w1.weight for layer in model.layers],
        *[layer.feed_forward.w2.weight for layer in model.layers],
        *[layer.feed_forward.w3.weight for layer in model.layers],
    ]
    if not shared_classifier:
        weights.append(model.output.weight)
    for w in weights:
        serialize_fp32(out_file, w)

    out_file.close()
    print(f"wrote {filepath}")
