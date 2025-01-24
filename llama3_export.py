import torch
import struct
import numpy as np
from llama3_model import Llama3Transformer, Llama3ModelArgs


def serialize_fp32(file, tensor):
    d = tensor.detach().cpu().view(-1).to(torch.float32).numpy()
    b = struct.pack(f"{len(d)}f", *d)
    file.write(b)


def serialize_int8(file, tensor):
    d = tensor.detach().cpu().view(-1).numpy().astype(np.int8)
    b = struct.pack(f"{len(d)}b", *d)
    file.write(b)


def quantize_q80(w, group_size):
    assert w.numel() % group_size == 0
    w = w.float().reshape(-1, group_size)
    wmax = torch.abs(w).max(dim=1).values
    scale = wmax / 127.0
    quant = w / scale[:, None]
    int8val = torch.round(quant).to(torch.int8)
    fp32val = (int8val.float() * scale[:, None]).view(-1)
    err = torch.abs(fp32val.reshape(-1, group_size) - w).max(dim=1).values
    return int8val, scale, err.max().item()


def version1_export(model, filepath):
    out_file = open(filepath, "wb")
    out_file.write(struct.pack("I", 0x616B3432))
    out_file.write(struct.pack("i", 1))
    p = model.params
    header = struct.pack(
        "iiiiiii",
        p.dim,
        p.hidden_dim,
        p.n_layers,
        p.n_heads,
        p.n_kv_heads,
        p.vocab_size,
        p.max_seq_len,
    )
    out_file.write(header)
    shared_classifier = torch.equal(
        model.tok_embeddings.weight, model.output.weight)
    out_file.write(struct.pack("B", int(shared_classifier)))
    pad = 256 - out_file.tell()
    out_file.write(b"\0" * pad)

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
    print(f"Exported model to {filepath}")


def version2_export(model, filepath, group_size=64):
    out_file = open(filepath, "wb")
    out_file.write(struct.pack("I", 0x616B3432))
    out_file.write(struct.pack("i", 2))
    p = model.params
    header = struct.pack(
        "iiiiiii",
        p.dim,
        p.hidden_dim,
        p.n_layers,
        p.n_heads,
        p.n_kv_heads,
        p.vocab_size,
        p.max_seq_len,
    )
    out_file.write(header)
    shared_classifier = torch.equal(
        model.tok_embeddings.weight, model.output.weight)
    out_file.write(struct.pack("B", int(shared_classifier)))
    out_file.write(struct.pack("i", group_size))
    pad = 256 - out_file.tell()
    out_file.write(b"\0" * pad)

    for layer in model.layers:
        serialize_fp32(out_file, layer.attention_norm.weight)
    for layer in model.layers:
        serialize_fp32(out_file, layer.ffn_norm.weight)
    serialize_fp32(out_file, model.norm.weight)

    weights = [
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

    for i, w in enumerate(weights):
        q, s, err = quantize_q80(w, group_size)
        serialize_int8(out_file, q)
        serialize_fp32(out_file, s)
        print(
            f"{i+1}/{len(weights)} quantized {tuple(w.shape)} to Q8_0 with max error {err}")

    out_file.close()
    print(f"Exported model to {filepath}")


def model_export(model, filepath, version):
    if version == 1:
        version1_export(model, filepath)
    elif version == 2:
        version2_export(model, filepath)
    else:
        raise ValueError(f"Unknown version {version}")
