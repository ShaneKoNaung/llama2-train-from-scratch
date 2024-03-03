# llama2-train-from-scratch
Train a LLamA2 model from scratch and convert it into GGUF format

The goal of this project is to learn about Llama2 architecture from Facebook, run it, train a very simple model locally, and then get it to run inside the llama.cpp framework.


## Setup

### prepare virtual env
#### uv install

```
# On macOS and Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows.
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# With pip.
pip install uv

# With pipx.
pipx install uv

# With Homebrew.
brew install uv

# With Pacman.
pacman -S uv
```

#### create venv
```
uv venv

source .venv/bin/activate
```

### install packages

```
# meta Llama2
uv pip install -e.
```



## Prepare dataset

### Tiny Shakespeare


```

python tinyshakespeare.py prepare

# output
Example story :
 ['First Citizen:', 'Before we proceed any further, hear me speak.', 'All:', 'Speak, speak.', 'First Citizen:', 'You are all resolved rather to die than to famish?', 'All:', 'Resolved. resolved.', 'First Citizen:', 'First, you know Caius Marcius is chief enemy to the people.']

```


### Prachathai67k

```
python prepare_thai_dataset.py download
python prepare_thai_dataset.py prepare
```



I have trained two different models using two different datasets. 
Datasets :
1. Tiny Shakespeare
2. prachathai67k


## Training a Model

Most of the code are from llama2.c. 

- download dataset
- prepare dataset
- tokenize dataset
- tokenizer model (Llama2 tokenizer)
- model parameter

## Converting to other formats
- export to llama2 .bin
- export to ggml
- export to gguf

## Text Generation
- text generation in llama2.c 
- text generation in pytorch llama
- text generation in ggml format
- text generation in gguf format
