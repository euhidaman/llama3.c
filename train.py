import os
import time
import torch
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from model import Transformer, ModelArgs
from tinystories import Task
from export import model_export

# Training configuration
out_dir = "outmini"
batch_size = 128
max_seq_len = 512
gradient_accumulation_steps = 1
vocab_source = "custom"
vocab_size = 512
dim = 64
n_layers = 5
n_heads = 8
n_kv_heads = 4
multiple_of = 4
learning_rate = 1e-3
dropout = 0.05
weight_decay = 0.01
max_iters = 100000
beta2 = 0.99
warmup_iters = 1000
eval_interval = 2000
eval_iters = 100
compile = True

# Model initialization
model_args = ModelArgs(
    dim=dim,
    n_layers=n_layers,
    n_heads=n_heads,
    n_kv_heads=n_kv_heads,
    vocab_size=vocab_size,
    multiple_of=multiple_of,
    max_seq_len=max_seq_len,
    dropout=dropout,
)
model = Transformer(model_args)
model.to("cuda")

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(
), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, beta2))

# Training loop
for iter_num in range(max_iters):
    # Training step
    x, y = next(Task.iter_batches(batch_size, "cuda", max_seq_len=max_seq_len,
                vocab_size=vocab_size, vocab_source=vocab_source))
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Evaluation
    if iter_num % eval_interval == 0:
        val_loss = estimate_loss()
        print(
            f"Iter {iter_num}: Train loss {loss.item():.4f}, Val loss {val_loss:.4f}")

    # Save checkpoint
    if iter_num % eval_interval == 0:
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iter_num": iter_num,
            "best_val_loss": val_loss,
        }
        torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
        model_export(model, os.path.join(out_dir, "model.bin"), version=1)
