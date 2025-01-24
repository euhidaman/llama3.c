import os
import json
import glob
import numpy as np
import torch
from torch.utils.data import IterableDataset
from tokenizer import Tokenizer
import sentencepiece as spm

DATA_CACHE_DIR = "data"

class PretokDataset(IterableDataset):
    def __init__(self, split, max_seq_len, vocab_size, vocab_source):
        self.split = split
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.vocab_source = vocab_source

    def __iter__(self):
        shard_filenames = sorted(glob.glob(os.path.join(DATA_CACHE_DIR, "*.bin")))
        for shard in shard_filenames:
            m = np.memmap(shard, dtype=np.uint16, mode="r")
            num_batches = len(m) // self.max_seq_len
            for ix in range(num_batches):
                start = ix * self.max_seq_len
                end = start + self.max_seq_len + 1
                chunk = torch.from_numpy((m[start:end]).astype(np.int64))
                x = chunk[:-1]
                y = chunk[1:]
                yield x, y

def download():
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    data_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"
    data_filename = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data.tar.gz")
    if not os.path.exists(data_filename):
        os.system(f"wget {data_url} -O {data_filename}")
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    if not os.path.exists(data_dir):
        os.system(f"tar -xzf {data_filename} -C {DATA_CACHE_DIR}")

def train_vocab(vocab_size):
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    tiny_file = os.path.join(DATA_CACHE_DIR, "tiny.txt")
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    with open(tiny_file, "w", encoding="utf-8") as of:
        for shard in shard_filenames:
            with open(shard, "r") as f:
                data = json.load(f)
            for example in data:
                text = example["story"].strip()
                of.write(text + "\n")
    prefix = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}")
    spm.SentencePieceTrainer.train(
        input=tiny_file,
        model_prefix=prefix,
        model_type="bpe",
        vocab_size=vocab_size,
        self_test_sample_size=0,
        input_format="text",
        character_coverage=1.0,
        num_threads=os.cpu_count(),
        split_digits=True,
        allow_whitespace_only_pieces=True,
        byte_fallback=True,
        unk_surface=r" \342\201\207 ",
        normalization_rule_name="identity",
    )
    print(f"Trained tokenizer is in {prefix}.model")

def pretokenize(vocab_size):
    bin_dir = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}")
    os.makedirs(bin_dir, exist_ok=True)
    shard_filenames = sorted(glob.glob(os.path.join(DATA_CACHE_DIR, "TinyStories_all_data", "*.json")))
    for shard in shard_filenames:
        with open(shard, "r") as f:
            data = json.load(f)
        all_tokens = []
        for example in data:
            text = example["story"].strip()
            tokens = Tokenizer().encode(text, bos=True, eos=False)
            all_tokens.extend(tokens)
        all_tokens = np.array(all_tokens, dtype=np.uint16)
        tokenized_filename = os.path.join(bin_dir, os.path.basename(shard).replace(".json", ".bin"))
        with open(tokenized_filename, "wb") as f:
            f.write(all_tokens.tobytes())

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=str, choices=["download", "train_vocab", "pretokenize"])
    parser.add_argument("--vocab_size", type=int, default=0, help="pretokenization vocab size. 0 = use Llama 2 tokenizer.")
    args = parser.parse_args()

    if args.stage == "download":
        download()
    elif args.stage == "train_vocab":
        train_vocab(vocab_size=args.vocab_size)
    elif args.stage == "pretokenize":
        pretokenize(vocab_size=args.vocab_size)
    else:
        raise ValueError(f"Unknown stage {args.stage}")