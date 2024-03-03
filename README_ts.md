# llama2-train-from-scratch
Train a LLamA2 model from scratch and convert it into GGUF format

The goal of this project is to learn about Llama2 architecture from Facebook, run it, train a very simple model locally, and then get it to run inside the llama.cpp framework.


## Setup

## Dataset

### Tiny Shakespeare

#### Prepare dataset

tiny shakespeare is a small dataset. The dataset is stored in `data/tinyshakespeare/input.txt`.


```
python tinyshakespeare.py prepare

# output
Example story :
 ['First Citizen:', 'Before we proceed any further, hear me speak.', 'All:', 'Speak, speak.', 'First Citizen:', 'You are all resolved rather to die than to famish?', 'All:', 'Resolved. resolved.', 'First Citizen:', 'First, you know Caius Marcius is chief enemy to the people.']
```

This step is just to load the dataset and look at it a bit. 




#### Tokenization


```
python tinyshakespeare.py pretokenize

# output
Saved data/tinyshakespeare/data.bin, average seqlen: 11.9347

```

Pretokenize stage uses the tokenizer model from Llama2-meta.
`SentencePieceProcessor` from `sentencepiece` is used for tokenizer model.

```
t = Tokenizer()
print(f"Encode : {t.encode('hello world', 1, 1)}")
print(f"Decode : {t.decode(t.encode('hello world',1, 1))}")

# output

Encode : [1, 22172, 3186, 2]
Decode : hello world
```

All the sentences in the dataset are tokenized and then saved in a `.bin` format. 

## Model training 

The models are trained on colab using this [notebook](workflow_colab.ipynb) for tinyshakespeares dataset and this [notebook](workflow_thai_colab.ipynb) for Thai dataset.

For model training, train.py from [llama2.c](https://github.com/karpathy/llama2.c) by [Andrej Karpathy](https://github.com/karpathy) is modified for our dataset. 

The main differences are 
    - `torch.nn.parallel.DistributedDataParallel` related codes are removed since we are only training a small dataset on a single GPU or on CPU
    - some `config` related codes are also removed because I wanted to limit the number of scripts needed for this repo. So, you have to hard code the configs in the train scripts.

Train script for tiny shakespeare : [train](train_tinyshakespeare.py).
Train script for Prachathai67k  : [train](train_thai.py)

Model parameters are : 

```

dim = 288
n_layers = 6
n_heads = 6
n_kv_heads = 6
multiple_of = 32
dropout = 0.0

```

## Model Export

### Export to llama2.c format

```
from export import model_export

model_export(model, os.path.join(out_dir, "model.bin"), version=0)
```

### Export to llama.cpp GGML format 

```
./convert-llama2c-to-ggml --copy-vocab-from-model ggml-vocab-llama.gguf --llama2c-model out_thai/model.bin --llama2c-output-model out_thai/llama2c-ggml.bin
```


### Export to llama.cpp GGUF format

```
python convert.py out_thai/ --ctx 4096

```

## Models

Models are under folders starting with out*.

out
├── ckpt.pt
├── ggml-model-f32.gguf
├── llama2c-ggml.bin
├── model.bin
└── params.json
out_thai
├── ckpt.pt
├── ggml-model-f32.gguf
├── llama2c-ggml.bin
└── model.bin


## Text Generation

### Llama2 pytorch
```
python generate_text.py

#output
compiling the model... (takes a ~minute)
LADY ANNE: Set down, set down your honourable load, If honour may be shrouded in a hearse, Whilst I awhile obsequiously lament The untimely fall of virtuous Lancaster. Poor key-cold figure of a holy king! Pale ashes of the house of Lancaster! Thou bloodless remnant of that royal blood! Be it lawful that I invocate thy ghost, To hear the lamentations of Poor Anne, Wife to thy Edward, to thy slaughter'd son, Stabb'd by the selfsame hand that made these wounds! Lo, in these windows that let forth thy life, I pour the helpless balm of my poor eyes. Cursed be the hand that made these fatal holes! Cursed be the heart that had the heart to do it! Cursed the blood that let this blood from hence! More direful hap betide that hated wretch, That makes us wretched by the
---------------
```

### Llama2.c bin

```
./run out/model.bin

#output
And Romeo dead; and Juliet, dead before,
</s>

achieved tok/s: 76.023392
```

### llama2.c bin to ggml

```
./main -m out/llama2c-ggml.bin -n 256 -t 0.7 --log-disable

Log start
main: build = 2295 (87c91c07)
main: built with cc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 for x86_64-linux-gnu
main: seed  = 1709500303
llama_model_loader: loaded meta data with 18 key-value pairs and 57 tensors from out/llama2c-ggml.bin (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                      tokenizer.ggml.tokens arr[str,32000]   = ["<unk>", "<s>", "</s>", "<0x00>", "<...
llama_model_loader: - kv   1:                      tokenizer.ggml.scores arr[f32,32000]   = [0.000000, 0.000000, 0.000000, 0.0000...
llama_model_loader: - kv   2:                  tokenizer.ggml.token_type arr[i32,32000]   = [2, 3, 3, 6, 6, 6, 6, 6, 6, 6, 6, 6, ...
llama_model_loader: - kv   3:                       tokenizer.ggml.model str              = llama
llama_model_loader: - kv   4:                       general.architecture str              = llama
llama_model_loader: - kv   5:                               general.name str              = llama
llama_model_loader: - kv   6:            tokenizer.ggml.unknown_token_id u32              = 0
llama_model_loader: - kv   7:                tokenizer.ggml.bos_token_id u32              = 1
llama_model_loader: - kv   8:                tokenizer.ggml.eos_token_id u32              = 2
llama_model_loader: - kv   9:          tokenizer.ggml.seperator_token_id u32              = 4294967295
llama_model_loader: - kv  10:            tokenizer.ggml.padding_token_id u32              = 4294967295
llama_model_loader: - kv  11:                       llama.context_length u32              = 128
llama_model_loader: - kv  12:                     llama.embedding_length u32              = 288
llama_model_loader: - kv  13:                  llama.feed_forward_length u32              = 768
llama_model_loader: - kv  14:                 llama.attention.head_count u32              = 6
llama_model_loader: - kv  15:                          llama.block_count u32              = 6
llama_model_loader: - kv  16:                 llama.rope.dimension_count u32              = 48
llama_model_loader: - kv  17:     llama.attention.layer_norm_rms_epsilon f32              = 0.000010
llama_model_loader: - type  f32:   57 tensors
llm_load_vocab: bad special token: 'tokenizer.ggml.seperator_token_id' = 4294967295d, using default id -1
llm_load_vocab: bad special token: 'tokenizer.ggml.padding_token_id' = 4294967295d, using default id -1
llm_load_vocab: special tokens definition check successful ( 259/32000 ).
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = llama
llm_load_print_meta: vocab type       = SPM
llm_load_print_meta: n_vocab          = 32000
llm_load_print_meta: n_merges         = 0
llm_load_print_meta: n_ctx_train      = 128
llm_load_print_meta: n_embd           = 288
llm_load_print_meta: n_head           = 6
llm_load_print_meta: n_head_kv        = 6
llm_load_print_meta: n_layer          = 6
llm_load_print_meta: n_rot            = 48
llm_load_print_meta: n_embd_head_k    = 48
llm_load_print_meta: n_embd_head_v    = 48
llm_load_print_meta: n_gqa            = 1
llm_load_print_meta: n_embd_k_gqa     = 288
llm_load_print_meta: n_embd_v_gqa     = 288
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-05
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: n_ff             = 768
llm_load_print_meta: n_expert         = 0
llm_load_print_meta: n_expert_used    = 0
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 0
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 10000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_yarn_orig_ctx  = 128
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: model type       = ?B
llm_load_print_meta: model ftype      = all F32 (guessed)
llm_load_print_meta: model params     = 24.41 M
llm_load_print_meta: model size       = 93.11 MiB (32.00 BPW)
llm_load_print_meta: general.name     = llama
llm_load_print_meta: BOS token        = 1 '<s>'
llm_load_print_meta: EOS token        = 2 '</s>'
llm_load_print_meta: UNK token        = 0 '<unk>'
llm_load_print_meta: LF token         = 13 '<0x0A>'
llm_load_tensors: ggml ctx size =    0.02 MiB
llm_load_tensors:        CPU buffer size =    93.11 MiB
...........................
llama_new_context_with_model: n_ctx      = 512
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:        CPU KV buffer size =     3.38 MiB
llama_new_context_with_model: KV self size  =    3.38 MiB, K (f16):    1.69 MiB, V (f16):    1.69 MiB
llama_new_context_with_model:        CPU input buffer size   =     2.57 MiB
llama_new_context_with_model:        CPU compute buffer size =    63.06 MiB
llama_new_context_with_model: graph splits (measure): 1
main: warning: model was trained on only 128 context tokens (512 specified)

system_info: n_threads = 8 / 8 | AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | FMA = 1 | NEON = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 |
sampling:
        repeat_last_n = 64, repeat_penalty = 1.100, frequency_penalty = 0.000, presence_penalty = 0.000
        top_k = 40, tfs_z = 1.000, top_p = 0.950, min_p = 0.050, typical_p = 1.000, temp = 0.800
        mirostat = 0, mirostat_lr = 0.100, mirostat_ent = 5.000
sampling order:
CFG -> Penalties -> top_k -> tfs_z -> typical_p -> top_p -> min_p -> temperature
generate: n_ctx = 512, n_batch = 512, n_predict = 256, n_keep = 1


 And at that sight shall sad Apollo weep, [end of text]

llama_print_timings:        load time =      23.73 ms
llama_print_timings:      sample time =       3.02 ms /    12 runs   (    0.25 ms per token,  3976.14 tokens per second)
llama_print_timings: prompt eval time =       0.00 ms /     1 tokens (    0.00 ms per token,      inf tokens per second)
llama_print_timings:        eval time =      49.52 ms /    12 runs   (    4.13 ms per token,   242.32 tokens per second)
llama_print_timings:       total time =      58.27 ms /    13 tokens
Log end
```

### llama.cpp GGUF

```
./main -m out/ggml-model-f32.gguf -n 256 --temp 0.7 

Log start
main: build = 2295 (87c91c07)
main: built with cc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 for x86_64-linux-gnu
main: seed  = 1709500389
llama_model_loader: loaded meta data with 15 key-value pairs and 57 tensors from out/ggml-model-f32.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = llama
llama_model_loader: - kv   1:                               general.name str              = LLaMA v2
llama_model_loader: - kv   2:                       llama.context_length u32              = 4096
llama_model_loader: - kv   3:                     llama.embedding_length u32              = 288
llama_model_loader: - kv   4:                          llama.block_count u32              = 6
llama_model_loader: - kv   5:                  llama.feed_forward_length u32              = 768
llama_model_loader: - kv   6:                 llama.rope.dimension_count u32              = 144
llama_model_loader: - kv   7:                 llama.attention.head_count u32              = 2
llama_model_loader: - kv   8:              llama.attention.head_count_kv u32              = 2
llama_model_loader: - kv   9:     llama.attention.layer_norm_rms_epsilon f32              = 0.000010
llama_model_loader: - kv  10:                          general.file_type u32              = 0
llama_model_loader: - kv  11:                       tokenizer.ggml.model str              = llama
llama_model_loader: - kv  12:                      tokenizer.ggml.tokens arr[str,32000]   = ["<unk>", "<s>", "</s>", "<0x00>", "<...
llama_model_loader: - kv  13:                      tokenizer.ggml.scores arr[f32,32000]   = [0.000000, 0.000000, 0.000000, 0.0000...
llama_model_loader: - kv  14:                  tokenizer.ggml.token_type arr[i32,32000]   = [2, 3, 3, 6, 6, 6, 6, 6, 6, 6, 6, 6, ...
llama_model_loader: - type  f32:   57 tensors
llm_load_vocab: special tokens definition check successful ( 259/32000 ).
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = llama
llm_load_print_meta: vocab type       = SPM
llm_load_print_meta: n_vocab          = 32000
llm_load_print_meta: n_merges         = 0
llm_load_print_meta: n_ctx_train      = 4096
llm_load_print_meta: n_embd           = 288
llm_load_print_meta: n_head           = 2
llm_load_print_meta: n_head_kv        = 2
llm_load_print_meta: n_layer          = 6
llm_load_print_meta: n_rot            = 144
llm_load_print_meta: n_embd_head_k    = 144
llm_load_print_meta: n_embd_head_v    = 144
llm_load_print_meta: n_gqa            = 1
llm_load_print_meta: n_embd_k_gqa     = 288
llm_load_print_meta: n_embd_v_gqa     = 288
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-05
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: n_ff             = 768
llm_load_print_meta: n_expert         = 0
llm_load_print_meta: n_expert_used    = 0
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 0
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 10000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_yarn_orig_ctx  = 4096
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: model type       = ?B
llm_load_print_meta: model ftype      = all F32
llm_load_print_meta: model params     = 24.41 M
llm_load_print_meta: model size       = 93.11 MiB (32.00 BPW)
llm_load_print_meta: general.name     = LLaMA v2
llm_load_print_meta: BOS token        = 1 '<s>'
llm_load_print_meta: EOS token        = 2 '</s>'
llm_load_print_meta: UNK token        = 0 '<unk>'
llm_load_print_meta: LF token         = 13 '<0x0A>'
llm_load_tensors: ggml ctx size =    0.02 MiB
llm_load_tensors:        CPU buffer size =    93.11 MiB
...........................
llama_new_context_with_model: n_ctx      = 512
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:        CPU KV buffer size =     3.38 MiB
llama_new_context_with_model: KV self size  =    3.38 MiB, K (f16):    1.69 MiB, V (f16):    1.69 MiB
llama_new_context_with_model:        CPU input buffer size   =     2.57 MiB
llama_new_context_with_model:        CPU compute buffer size =    63.06 MiB
llama_new_context_with_model: graph splits (measure): 1

system_info: n_threads = 4 / 8 | AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | FMA = 1 | NEON = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 |
sampling:                                                                                                                                                                                                                                                                                                                                                                                        repeat_last_n = 64, repeat_penalty = 1.100, frequency_penalty = 0.000, presence_penalty = 0.000
        top_k = 40, tfs_z = 1.000, top_p = 0.950, min_p = 0.050, typical_p = 1.000, temp = 0.700
        mirostat = 0, mirostat_lr = 0.100, mirostat_ent = 5.000
sampling order:
CFG -> Penalties -> top_k -> tfs_z -> typical_p -> top_p -> min_p -> temperature
generate: n_ctx = 512, n_batch = 512, n_predict = 256, n_keep = 1


 And, like a twod- JULIET: hast nothing else, com ill-hearted. [end of text]

llama_print_timings:        load time =      11.72 ms
llama_print_timings:      sample time =       6.23 ms /    23 runs   (    0.27 ms per token,  3689.44 tokens per second)
llama_print_timings: prompt eval time =       0.00 ms /     1 tokens (    0.00 ms per token,      inf tokens per second)
llama_print_timings:        eval time =      69.67 ms /    23 runs   (    3.03 ms per token,   330.13 tokens per second)
llama_print_timings:       total time =      87.30 ms /    24 tokens
Log end
```