import argparse
import os
import numpy as np
import random
import glob

import torch
import torch.distributed as dist

from functools import partial

from tokenizer import Tokenizer

DATA_CACHE_DIR = "data"
data_dir = os.path.join(DATA_CACHE_DIR, 'tinyshakespeare')

def prepare():
    
    data_dir = os.path.join(DATA_CACHE_DIR, 'tinyshakespeare')
    
    filename = os.path.join(data_dir, 'input.txt')
    data = []
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(line)
    print(f"Example story : \n {data[:10]}")


def pretokenize():
    data_dir = os.path.join(DATA_CACHE_DIR, "tinyshakespeare")
    enc = Tokenizer()
    
    filename = os.path.join(data_dir, "input.txt")
    all_tokens = []
    with open(filename, "r") as f:
        for text in f:
            text = text.strip()
            if text:
                tokens = enc.encode(text, bos=True, eos=True)
                all_tokens.extend(tokens)
    all_tokens = np.array(all_tokens, dtype=np.uint16)
    
    tokenized_filename = os.path.join(data_dir, "data.bin")
    
    with open(tokenized_filename, "wb") as f:
        f.write(all_tokens.tobytes())
    
    avg_seq_len = all_tokens.size / ((all_tokens == 1).sum())
    print(f"Saved {tokenized_filename}, average seqlen: {avg_seq_len:.4f}")

class PretokDataset(torch.utils.data.IterableDataset):
    def __init__(self, split, max_seq_len, vocab_size):
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = 42 + worker_id + 1337 * rank
        rng = random.Random(seed)
        print(f"Created a PretokDataset with rng seed {seed}")
        
        bin_dir = os.path.join(DATA_CACHE_DIR, "tinyshakespeare")
        filename = sorted(glob.glob(os.path.join(bin_dir, "*.bin")))[0]
        
        assert len(filename)>0, f"No bin files found in {bin_dir}"
        while True:
                
            m = np.memmap(filename, dtype=np.uint16, mode="r")
            num_batches = len(m) // self.max_seq_len
            num_batches -= 1
            assert num_batches > 0, "this file is way too small? investigatte."
            ixs = list(range(num_batches))
            rng.shuffle(ixs)
            for ix in ixs:
                start = ix * self.max_seq_len
                end = start + self.max_seq_len + 1
                chunk = torch.from_numpy((m[start: end]).astype(np.int64))
                x = chunk[:-1]
                y = chunk[1:]
                yield x,y


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


# -----------------------------------------------------------------------------
# CLI for constructing the dataset

if __name__ == "__main__":
    """
    These stages are designed to be run in order.

    To tokenize data with the Llama 2 tokenizer:
    python tinystories.py prepare
    python tinystories.py pretokenize

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=str, choices=["prepare", "pretokenize"])
    args = parser.parse_args()

    # depending on the stage call the appropriate function
    if args.stage == "prepare":
        prepare()
    elif args.stage == "pretokenize":
        pretokenize()
    else:
        raise ValueError(f"Unknown stage {args.stage}")