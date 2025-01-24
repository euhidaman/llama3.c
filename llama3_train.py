import math
import os
import time
from contextlib import nullcontext
from datetime import datetime
from functools import partial

import torch
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from llama3_model import Llama3Transformer, Llama3ModelArgs
from tinystories import Task
from llama3_export import model_export

out_dir = "out"
eval_interval = 2000
log_interval = 1
eval_iters = 100
eval_only = False
always_save_checkpoint = True
init_from = "scratch"
batch_size = 128
max_seq_len = 256
vocab_source = "llama3"
vocab_size = 32000

dim = 4096
n_layers = 32
n_heads = 32
n_kv_heads = 32
multiple_of = 256
dropout = 0.1

gradient_accumulation_steps = 4
learning_rate = 5e-4
max_iters = 100000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

decay_lr = True
warmup_iters = 1000
lr_decay_iters = max_iters
min_lr = 0.0

device = "cuda"
dtype = "float32"
compile = True

model_args = Llama3ModelArgs(
    dim=dim,
    n_layers=n_layers,
    n_heads=n_heads,
    n_kv_heads=n_kv_heads,
    vocab_size=vocab_size,
    multiple_of=multiple_of,
    max_seq_len=max_seq_len,
    dropout=dropout,
)

if init_from == "scratch":
    model = Llama3Transformer(model_args)
elif init_from == "resume":
    checkpoint = torch.load(os.path.join(
        out_dir, "ckpt.pt"), map_location=device)
    model = Llama3Transformer(model_args)
    model.load_state_dict(checkpoint["model"])

model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(
    beta1, beta2), weight_decay=weight_decay)

if compile:
    model = torch.compile(model)

ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    init_process_group(backend="nccl")
    model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])])

task = Task()
train_loader = task.iter_batches(
    batch_size=batch_size, device=device, max_seq_len=max_seq_len, vocab_source=vocab_source)

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

for iter_num in range(max_iters):
    if iter_num % eval_interval == 0:
        losses = {}
        model.eval()
        for split in ["train", "val"]:
            batch_iter = task.iter_batches(
                split=split, batch_size=batch_size, device=device, max_seq_len=max_seq_len)
            losses[split] = torch.zeros(eval_iters)
            for k in range(eval_iters):
                x, y = next(batch_iter)
                with torch.amp.autocast(device_type=device, dtype=torch.bfloat16 if dtype == "bfloat16" else torch.float16 if dtype == "float16" else torch.float32):
                    logits, loss = model(x, y)
                losses[split][k] = loss.item()
            losses[split] = losses[split].mean()
        model.train()
        print(
            f"Step {iter_num}: Train loss {losses['train']:.4f}, Val loss {losses['val']:.4f}")
        if losses["val"] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses["val"]
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "model_args": model_args,
                "iter_num": iter_num,
                "best_val_loss": best_val_loss,
            }
            torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
            model_export(model, os.path.join(out_dir, "model.bin"), version=1)

    x, y = next(train_loader)
    with torch.amp.autocast(device_type=device, dtype=torch.bfloat16 if dtype == "bfloat16" else torch.float16 if dtype == "float16" else torch.float32):
        logits, loss = model(x, y)
    scaler.scale(loss).backward()

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()

    if iter_num % log_interval == 0:
        print(f"Iter {iter_num}: Loss {loss.item():.4f}")

if ddp:
    destroy_process_group()
