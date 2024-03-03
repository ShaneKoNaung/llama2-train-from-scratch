import os
import time
import random
import glob

import torch
import torch.distributed as dist

from contextlib import nullcontext
from functools import partial

from model import Transformer, ModelArgs
from tokenizer import Tokenizer


#-----------configs----------
out_dir = "out"
start = ""
num_samples = 1 # number of samples to draw
max_new_tokens = 256 # number of tokens generated in each sample
temperature = 0.7 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 300


device = 'cpu'
device_type = "cuda" if "cuda" in device else "cpu"
#--------------------------

# model
dim = 288
n_layers = 6
n_heads = 6
n_kv_heads = 6
multiple_of = 32
dropout = 0.0


vocab_size = 32000
max_seq_len = 512

model_args = dict(
    dim=dim,
    n_layers=n_layers,
    n_heads=n_heads,
    n_kv_heads=n_kv_heads,
    vocab_size=vocab_size,
    multiple_of=multiple_of,
    max_seq_len=max_seq_len,
    dropout=dropout,
)


ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)

ckpt_path = os.path.join(out_dir, "ckpt.pt")
checkpoint = torch.load(ckpt_path, map_location=device)
checkpoint_model_args = checkpoint["model_args"]
# force these config attributes to be equal otherwise we can't even resume training
# the rest of the attributes (e.g. dropout) can stay as desired from command line
for k in ["dim", "n_layers", "n_heads", "n_kv_heads", "vocab_size", "multiple_of", "max_seq_len"]:
    model_args[k] = checkpoint_model_args[k]
# create the model
gptconf = ModelArgs(**model_args)
model = Transformer(gptconf)
state_dict = checkpoint["model"]

# fix the keys of the state dictionary :(
# honestly no idea how checkpoints sometimes get this prefix, have to debug more
unwanted_prefix = "_orig_mod."
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
model.load_state_dict(state_dict)


# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)  # requires PyTorch 2.0

model.eval()
model.to(device)



t = Tokenizer()

start_ids = t.encode(start, bos=True, eos=False)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(t.decode(y[0].tolist()))
            print('---------------')