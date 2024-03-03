import argparse
import os
import numpy as np
import random
import glob
import json

import torch
import torch.distributed as dist

from functools import partial

from tokenizer import Tokenizer

DATA_CACHE_DIR = "data"
data_dir = os.path.join(DATA_CACHE_DIR, 'prachathai67k')

def download_file(url: str, fname: str, chunk_size=1024):
    
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
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    
    data_url = "https://archive.org/download/prachathai67k/data.zip"
    data_filename = os.path.join(DATA_CACHE_DIR, "data.zip")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)
    else:
        print(f"{data_filename} already exists, skipping download...")
    
    data_dir = os.path.join(DATA_CACHE_DIR, "prachathai67k")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        print(f"Unpacking {data_filename}...")
        os.system(f"unzip {data_filename} -d {data_dir}")
        os.system(f"mv {data_dir}/data/* {data_dir}")
        os.system(f"rm -rf {data_dir}/data/")
    else:
        print(f"{data_dir} already exists, skipping unpacking...")
    
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.jsonl")))
    data = []
    with open(shard_filenames[0], "r") as f:
        for line in f:
            data.append(json.loads(line))
            
    print("Download done.")
    print(f"Number of shards: {len(shard_filenames)}")
    print(f"Example story:\n{data[0]}")

def prepare():
    '''
    combine train.jsonl and valid.jsonl and convert all three jsonl to

    <title> <bodytext> <topic> ...
    '''
    data_dir = os.path.join(DATA_CACHE_DIR, "prachathai67k")
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.jsonl")))

    txt_filename = os.path.join(data_dir, 'train.txt')


    train = []
    test = []

    if not os.path.exists(txt_filename):

        for shard in shard_filenames:
            if 'train' in shard or 'valid' in shard:
                with open(shard, "r") as f:
                    for line in f:
                        data = json.loads(line)
                        text = [v for k,v in data.items() if k in ['title', 'body_text']]
                        text = [i.replace('\n', ' ') for i in text]
                        text.extend([k for k, v in data.items() if v == 1])
                        text = ' '.join(text)
                        text.replace('\n', ' ')
                        train.append(text)
            else:
                with open(shard, "r") as f:
                    for line in f:
                        data = json.loads(line)
                        text = [v for k,v in data.items() if k in ['title', 'body_text']]
                        text = [i.replace('\n', ' ') for i in text]
                        text.extend([k for k, v in data.items() if v == 1])
                        text = ' '.join(text)
                        text.replace('\n', ' ')
                        test.append(text)

        train_txt_filename = os.path.join(data_dir, 'train.txt')
        test_txt_filename  = os.path.join(data_dir, 'test.txt')
        with open(train_txt_filename, 'w') as f:
            for i in train:
                f.write(i + '\n')

        with open(test_txt_filename, 'w') as f:
            for i in test:
                f.write(i + '\n')
    else:
        train_txt_filename = os.path.join(data_dir, 'train.txt')
        test_txt_filename  = os.path.join(data_dir, 'test.txt')
        print(f"{txt_filename} already exists, skipping preparing...")

    train = []
    test = []
    with open(train_txt_filename, 'r') as f:
        for i in f:
            train.append(i.strip())

    with open(test_txt_filename, 'r') as f:
        for i in f:
            test.append(i.strip())


    print(f"Number of train samples : {len(train)}")
    print(f"Number of test samples : {len(test)}")

    print(f"Train sample : \n\t{train[0]}\n")
    print(f"Test sample : \n\t{test[0]}\n")


def pretokenize():
    data_dir = os.path.join(DATA_CACHE_DIR, "prachathai67k")
    enc = Tokenizer()

    filenames = sorted(glob.glob(os.path.join(data_dir, "*.txt")))


    for filename in filenames:
        txt_basename = os.path.basename(filename)
        bin_basename = txt_basename.replace('.txt', '.bin')
        tokenized_filename = os.path.join(data_dir, bin_basename)

        if not os.path.exists(tokenized_filename):

            all_tokens = []
            with open(filename, "r") as f:
                for text in f:
                    text = text.strip()
                    if text:
                        tokens = enc.encode(text, bos=True, eos=True)
                        all_tokens.extend(tokens)

            all_tokens = np.array(all_tokens, dtype=np.uint16)



            with open(tokenized_filename, "wb") as f:
                f.write(all_tokens.tobytes())

            avg_seq_len = all_tokens.size / ((all_tokens == 1).sum())
            print(f"Saved {tokenized_filename}, average seqlen: {avg_seq_len:.4f}")
        else:
            print(f"{tokenized_filename} exists,  pretokenization is already done for {txt_basename}")


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

        bin_dir = os.path.join(DATA_CACHE_DIR, "prachathai67k")
        filename = os.path.join(bin_dir, "train.bin") if self.split == "train" else os.path.join(bin_dir, "test.bin")

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
            ds, batch_size=batch_size, pin_memory=False, num_workers=num_workers
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
    parser.add_argument("stage", type=str, choices=["download", "prepare", "pretokenize"])
    args = parser.parse_args()

    # depending on the stage call the appropriate function
    if args.stage == "download":
        download()
    elif args.stage == "prepare":
        prepare()
    elif args.stage == "pretokenize":
        pretokenize()
    else:
        raise ValueError(f"Unknown stage {args.stage}")