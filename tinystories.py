import os
import glob
import json
import requests
import numpy as np
import torch
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tokenizer import Tokenizer
import random

DATA_CACHE_DIR = "data"


def download_file(url: str, fname: str, chunk_size=1024):
    """Helper function to download a file from a given url"""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


def download():
    """Downloads the TinyStories dataset to DATA_CACHE_DIR"""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    # Download the TinyStories dataset
    data_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"
    data_filename = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data.tar.gz")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)
    else:
        print(f"{data_filename} already exists, skipping download...")

    # Unpack the tar.gz file
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        print(f"Unpacking {data_filename}...")
        os.system(f"tar -xzf {data_filename} -C {data_dir}")
    else:
        print(f"{data_dir} already exists, skipping unpacking...")

    # Print a single example for debugging
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    with open(shard_filenames[0], "r") as f:
        data = json.load(f)
    print("Download done.")
    print(f"Number of shards: {len(shard_filenames)}")
    print(f"Example story:\n{data[0]}")


def process_shard(args):
    """Tokenizes a single shard of the dataset"""
    shard_id, shard = args
    enc = Tokenizer()  # Use the tiktoken tokenizer
    with open(shard, "r") as f:
        data = json.load(f)
    all_tokens = []
    for example in tqdm(data, position=shard_id):
        text = example["story"]
        text = text.strip()
        tokens = enc.encode(text, bos=True, eos=False)  # Encode with BOS token
        all_tokens.extend(tokens)
    all_tokens = np.array(all_tokens, dtype=np.uint16)
    tokenized_filename = shard.replace(".json", ".bin")
    with open(tokenized_filename, "wb") as f:
        f.write(all_tokens.tobytes())
    avg_seq_len = all_tokens.size / ((all_tokens == enc.bos_id).sum())
    print(f"Saved {tokenized_filename}, average seqlen: {avg_seq_len:.2f}")


def pretokenize():
    """Tokenizes all shards of the dataset"""
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    with ProcessPoolExecutor() as executor:
        executor.map(process_shard, enumerate(shard_filenames))
    print("Done.")


class PretokDataset(torch.utils.data.IterableDataset):
    """Loads pretokenized examples from disk and yields them as PyTorch tensors."""

    def __init__(self, split, max_seq_len, vocab_size, vocab_source):
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.vocab_source = vocab_source

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        seed = 42 + worker_id + 1337 * rank
        rng = random.Random(seed)
        print(f"Created a PretokDataset with rng seed {seed}")
        bin_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
        shard_filenames = sorted(glob.glob(os.path.join(bin_dir, "*.bin")))
        shard_filenames = shard_filenames[1:
                                          ] if self.split == "train" else shard_filenames[:1]
        assert len(shard_filenames) > 0, f"No bin files found in {bin_dir}"
        while True:
            rng.shuffle(shard_filenames)
            for shard in shard_filenames:
                m = np.memmap(shard, dtype=np.uint16, mode="r")
                num_batches = len(m) // self.max_seq_len
                num_batches -= 1
                assert num_batches > 0, "this shard is way too small? investigate."
                ixs = list(range(num_batches))
                rng.shuffle(ixs)
                for ix in ixs:
                    start = ix * self.max_seq_len
                    end = start + self.max_seq_len + 1
                    chunk = torch.from_numpy((m[start:end]).astype(np.int64))
                    x = chunk[:-1]
                    y = chunk[1:]
                    yield x, y


def get_tokenizer_model_path(vocab_size):
    """Returns None since we're using tiktoken"""
    return None


class Task:
    @staticmethod
    def iter_batches(batch_size, device, num_workers=0, **dataset_kwargs):
        ds = PretokDataset(**dataset_kwargs)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers
        )
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            yield x, y


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=str, choices=["download", "pretokenize"])
    args = parser.parse_args()

    if args.stage == "download":
        download()
    elif args.stage == "pretokenize":
        pretokenize()
    else:
        raise ValueError(f"Unknown stage {args.stage}")
