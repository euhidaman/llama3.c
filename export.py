import os
import struct
import torch
from model import Transformer, ModelArgs


def serialize_fp32(file, tensor):
    """Writes one fp32 tensor to a file that is open in write-binary mode."""
    d = tensor.detach().cpu().view(-1).to(torch.float32).numpy()
    b = struct.pack(f'{len(d)}f', *d)
    file.write(b)


def serialize_int8(file, tensor):
    """Writes one int8 tensor to a file that is open in write-binary mode."""
    d = tensor.detach().cpu().view(-1).numpy().astype(np.int8)
    b = struct.pack(f'{len(d)}b', *d)
    file.write(b)


def quantize_q80(w, group_size):
    """
    Quantizes a tensor to int8 (Q8_0 format) in groups of `group_size`.
    Returns the quantized tensor, scale factors, and maximum error.
    """
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
    """
    Exports the model weights in full float32 .bin file to be read from C.
    This is the same as legacy_export, but with a proper header.
    """
    version = 1

    out_file = open(filepath, 'wb')
    # First write out the header. The header will be 256 bytes
    # 1) write magic, which will be uint32 of "ak42" in ASCII
    out_file.write(struct.pack('I', 0x616b3432))
    # 2) write version, which will be int
    out_file.write(struct.pack('i', version))
    # 3) write the params, which will be 7 ints
    p = model.params
    hidden_dim = model.layers[0].feed_forward.w3.weight.shape[0]
    n_kv_heads = p.n_kv_heads
    header = struct.pack('iiiiiii', p.dim, hidden_dim, p.n_layers, p.n_heads,
                         n_kv_heads, p.vocab_size, p.max_seq_len)
    out_file.write(header)
    # 4) write some other flags
    shared_classifier = torch.equal(
        model.tok_embeddings.weight, model.output.weight)
    out_file.write(struct.pack('B', int(shared_classifier)))
    pad = 256 - out_file.tell()  # pad rest with zeros; tell returns current pos
    assert pad >= 0
    out_file.write(b'\0' * pad)

    # Now let's write out all the params
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

    # Write to binary file
    out_file.close()
    print(f"wrote {filepath}")


def model_export(model, filepath, version, dtype=torch.float32):
    """
    Exports the model to a .bin file.
    Versions:
    - v0: legacy export (DEPRECATED)
    - v1: float32 export
    - v2: int8 quantized Q8_0 export
    """
    if version == 0:
        raise ValueError("Legacy export (v0) is deprecated. Use v1 or v2.")
    elif version == 1:
        version1_export(model, filepath)
    elif version == 2:
        raise NotImplementedError("Q8_0 export (v2) is not yet implemented.")
    else:
        raise ValueError(f"Unknown version {version}")
