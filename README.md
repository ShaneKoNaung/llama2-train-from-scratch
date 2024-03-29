# llama2-train-from-scratch
Train a LLamA2 model from scratch and convert it into GGUF format

The goal of this project is to learn about Llama2 architecture from Facebook, run it, train a very simple model locally, and then get it to run inside the llama.cpp framework.


## Setup

## Dataset

### Prachathai67k

### Download dataset

```
python prepare_thai_data.py download

#output

data/data.zip already exists, skipping download...
Unpacking data/data.zip...
Archive:  data/data.zip
   creating: data/prachathai67k/data/
  inflating: data/prachathai67k/data/train.jsonl  
  inflating: data/prachathai67k/data/test.jsonl  
  inflating: data/prachathai67k/data/valid.jsonl  
Download done.
Number of shards: 3
Example story:
{'url': 'https://prachatai.com/print/62490', 'date': '2015-11-17 18:14', 'title': 'แฮคเกอร์ Anonymous ลั่นทำสงครามไซเบอร์ครั้งใหญ่สุดกับกลุ่ม IS', 'body_text': '17 พ.ย. 2558 Blognone [1] รายงานว่า กลุ่มแฮคเกอร์ Anonymous ประกาศสงครามไซเบอร์กับกลุ่มหัวรุนแรงหลังจากกลุ่ม IS ออกมาประกาศว่าเป็นผู้อยู่เบื้องหลังการโจมตีกรุงปารีสในคืนวันศุกร์ที่ผ่านมา\n\n\nภาพในคลิปใน YouTube โฆษกของกลุ่มแฮคเกอร์สวมหน้ากากที่เป็นสัญลักษณ์ของกลุ่มได้ออกมาอ่านแถลงเป็นภาษาฝรั่งเศส มีใจความว่า จากการโจมตีของกลุ่ม IS ในกรุงปารีส กลุ่ม Anonymous ทั่วโลกจะตามล่ากลุ่ม IS เหมือนที่เคยทำตอนที่มีการโจมตีสำนักพิมพ์ Charlie Hebdo และครั้งนี้จะเป็นปฏิบัติการโจมตีครั้งใหญ่ที่สุดของกลุ่ม Anonymous เลย นอกจากนี้กลุ่ม Anonymous ยังแสดงความเสียใจต่อครอบครัวผู้สูญเสียในเหตุการณ์ครั้งนี้\nกลุ่ม Anonymous เคยประกาศสงครามกับกลุ่ม IS หลังจากการโจมตีสำนักพิมพ์ Charlie Hebdo ที่ฝรั่งเศสเมื่อต้นปีที่ผ่านมา ซึ่งครั้งนั้นกลุ่ม Anonymous อ้างว่าได้ระงับบัญชีผู้ใช้งานที่เกี่ยวข้องกับ IS ไปหลายพันบัญชี (อ่านรายละเอียดเพิ่มเติม จากBlognone ที่\xa0\xa0กลุ่มแฮคเกอร์ Anonymous ประกาศสงครามไซเบอร์ขอกวาดล้างพวก ISIS [2])', 'politics': 0, 'human_rights': 0, 'quality_of_life': 0, 'international': 1, 'social': 0, 'environment': 0, 'economics': 0, 'culture': 0, 'labor': 0, 'national_security': 0, 'ict': 1, 'education': 0}
```

#### Prepare dataset

```
python prepare_thai_data.py prepare

# output
Number of train samples : 61100
Number of test samples : 6789
Train sample : 
	วุฒิสภาจี้หาทางออกเหมืองโปแตช ประชาไท9 ก.พ. 2549 เมื่อวันที่ 8 ก.พ. เวลาประมาณ 14.00 น. คณะกรรมาธิการสิ่งแวดล้อมวุฒิสภาได้จัดเวทีประชุมติดตามความคืบหน้าโครงการเหมืองแร่โปแตช จ. อุดรธานี เนื่องจากเห็นว่ามีการผลักดันโครงการอย่างเร่งด่วนโดยไม่คำนึงถึงการมีส่วนร่วมของประชาชนในพื้นที่ ณ ห้องประชุมกรรมาธิการหมายเลข 306 อาคารรัฐสภาตามข้อเรียกร้องของกลุ่มอนุรักษ์สิ่งแวดล้อมอุดรธานี โดยได้เชิญหน่วยงานที่เกี่ยวข้องได้แก่ กรมอุตสาหกรรมพื้นฐานและการเหมืองแร่ สำนักงานนโยบายและแผนทรัพยากรธรรมชาติสิ่งแวดล้อม กลุ่มอนุรักษ์สิ่งแวดล้อมอุดรธานี เข้ามาชี้แจงให้ข้อมูล      นายสุรพงษ์ เชียงทอง เจ้าหน้าที่กรมอุตสาหกรรมพื้นฐานและการเหมืองแร่ (กพร.) รายงานต่อที่ประชุมว่าขณะนี้บริษัทได้ยื่นขอประธานบัตรทำเหมืองแล้วอยู่ระหว่างการขึ้นรูปแผนที่พื้นที่ทำเหมืองใต้ดิน ในขณะนี้ทางบริษัทต้องทำการรังวัดปักหมดเขตที่ตั้งโรงแยกแร่ หรือเหมืองแร่บนดินและขึ้นรูปแผนที่เพื่อจะได้ติดประกาศในท้องถิ่นก่อนที่อธิบดีกพร.จะรับรอง แต่อย่างไรก็ตามขณะนี้บริษัยยังไม่ได้ยื่นเอกสารเพื่อการพิจารณาเข้ามาอย่างครบถ้วน และยังต้องดำเนินการอีกหลายขั้นตอนตามกฎหมายแร่ฉบับปี 2545        นายแก้วสรร อติโพธิ ประธานกรรมาธิการสิ่งแวดล้อมวุฒิสภา กล่าวว่าจาการที่เจ้าหน้าที่จากสำนักงานนโยบายและแผนทรัพยากรธรรมชาติและสิ่งแวดล้อม(สผ.)รายงานต่อที่ประชุมว่าคณะนี้บริษัทยังไม่ได้ส่งรายงานฉบับใหม่ที่ต้องมีรายละเอียดเรื่องการศึกษาวิเคราะห์ผลกระทบตามสารสำคัญของกฎหมายแร่ฉบับใหม่ที่ครอบคลุมการทำเหมืองแร่ใต้ดินให้ สผ.พิจารณา อย่างไรก็ตามในการพิจารณาของ สผ. ซึ่งเป็นหน่วยงานราชการมีอำนาจจะตั้งคำถามทางเทคนิควิธีการเพื่อสร้างทางเลือก เช่น การทำเหมืองแร่ใต้ดินแบบที่บริษัทเสนอเป็นแบบช่องทางสลับค้ำยันนี้ปลอดภัยจริงหรือไม่เมื่อเหมืองอยู่ใต้ชุมชน มีทางเลือกวิธีการทำเหมืองใต้ดินแบบที่ปลอดภัยกว่านี้เช่นแบบเหมืองละลายแร่ (Solution mining) เรื่อง กองหางแร่ไม่ต้องพิจารณาเพียงว่าจะใช้ผ้ายางหนาเท่าใดหรือมีผู้เชี่ยวชาญหรือไม่เมื่อนำเอาเกลือปริมาณมากกองบนผิวดินมันจะต้องมีผลกระทบแน่ ๆ ทางกรรมาธิการเป็นห่วงเรื่องนี้มาก จะต้องชี้แจงให้บริษัทแก้ไข การนำเกลือลงไปถมกลับนั้นเราก็รู้กันอยู่ว่าเกลือมีราคาและการนำกลับลงไปก็มีค่าใช้จ่าย และทั้ง สผ. และ กพร. ต่างมีความเห็นพ้องกันว่าสามารถนำไปใช้ในอุตสาหกรรมต่อเนื่องได้ บริษัทจะต้องชี้แจงให้ชัดเจนในรายงานว่าเกลือจะขายให้อุตสาหกรรมต่อเนื่องหรือจะถมกลับ        นายแก้วสรร ยังกล่าวอีกว่าแม้บริษัทจะยื่นขอประทานบัตรเป็นบริเวณกว้าง 2 แหล่งมีชุมชนหลายชุมชนตั้งอยู่ข้างบน กพร.ก็ไม่จำเป็นต้องอนุญาตทั้งหมดที่บริษัทขอ มันขึ้นอยู่ที่การพิจารณาของ กพร. จะเป็นไปได้หรือไม่ที่จะละเว้นเขตที่มีชุมชนตั้งอยู่ และทางจังหวัดจะต้องจัดทำแผนที่แสดงความหนาแน่นของประชากรให้ชัดเจนว่าเขตนี้มีประชากรอยู่เท่าไหร่ ถ้ามีประชากรอยู่ก็ละเว้นไม่อนุญาตให้ทำ เพราะหน่วยงานราชการต้องสร้างทางเลือกที่เหมาะสม  ลดความขัดแย้งเพราะถ้าสร้างเหมืองแร่บนความขัดแย้ง  ในชุมชนก็ไม่มีวันสงบ หากชุมชนไม่ยอมรับเหมือนกรณีโครงการท่อก๊าซไทยมาเลเซีย ที่สร้างบนความขัดแย้งปัจจุบันโครงการก็อยู่อย่างหวาดผวาว่าจะมีคนมาวางระเบิดเมื่อไร นายแก้วสรรกล่าว        ด้านพลเอกสมคิด ศรีสังคม สว.อุดรธานี กล่าวว่า สิ่งที่น่าห่วงหากเกิดโครงการคือเรื่องผลกระทบจากเกลือหางแร่เพราะภาคอีสานลมแรงในฤดูแล้ง พายุฤดูแล้งลมแรงมาก ขณะที่ฤดูฝนก็มีฝนมาก ที่ตั้งโครงการก็เช่นกันโรงแต่งแร่ตั้งอยู่บนเนินสูง กองเกลือกว้างเป็นกิโล และสูง 40 เมตรไม่ต้องใช้ผู้เชี่ยวชาญพิจารณาให้ลำบากก็เห็นว่าจะมีผลกระทบอยู่ชัด ๆ ในฐานะคนอุดรธานีผมไม่เห็นด้วยกับโครงการไม่อยากให้สร้างเหมืองแร่ในจังหวัดอุดรธานี        ด้านนางมณี บุญรอด  รองประธานกลุ่มอนุรักษ์สิ่งแวดล้อมอุดรธานี กล่าวชี้แจงว่าปัจจุบันชาวบ้านไม่ยอมรับโครงการและมีการทำงานแย่งแยกชาวบ้าน เกิดความขัดแย้งรุนแรงขึ้นเรื่อย ๆ แม้ยังไม่สร้างโครงการ คือสิ่งที่น่าเป็นห่วงที่สุด และหากโครงการยังดำรงอยู่ความขัดแย้งก็รุนแรงขึ้นเรื่อย ๆ แม้บริษัทจะอ้างว่าได้ลงทุนไปแล้วหลายพันล้าน รวมทั้งบริษัทอ้างว่าจะฟ้องร้องรัฐบาลไทยหากไม่ได้ทำเหมือง และจะเป็นการทำลายความเชื่อมั่นคงนักลงทุนต่างชาติ  นั้นเป็นการพูดแต่ได้เพราะบริษัทเองละเลยขั้นตอนกฎหมายไทย ละเมิดสิทธิคนไทย  และกำลังมีการยุแยงให้คนไทย คนในชุมชนเดียวกันขัดแย้งเข่นฆ่ากันเอง politics

Test sample : 
	แฮคเกอร์ Anonymous ลั่นทำสงครามไซเบอร์ครั้งใหญ่สุดกับกลุ่ม IS 17 พ.ย. 2558 Blognone [1] รายงานว่า กลุ่มแฮคเกอร์ Anonymous ประกาศสงครามไซเบอร์กับกลุ่มหัวรุนแรงหลังจากกลุ่ม IS ออกมาประกาศว่าเป็นผู้อยู่เบื้องหลังการโจมตีกรุงปารีสในคืนวันศุกร์ที่ผ่านมา   ภาพในคลิปใน YouTube โฆษกของกลุ่มแฮคเกอร์สวมหน้ากากที่เป็นสัญลักษณ์ของกลุ่มได้ออกมาอ่านแถลงเป็นภาษาฝรั่งเศส มีใจความว่า จากการโจมตีของกลุ่ม IS ในกรุงปารีส กลุ่ม Anonymous ทั่วโลกจะตามล่ากลุ่ม IS เหมือนที่เคยทำตอนที่มีการโจมตีสำนักพิมพ์ Charlie Hebdo และครั้งนี้จะเป็นปฏิบัติการโจมตีครั้งใหญ่ที่สุดของกลุ่ม Anonymous เลย นอกจากนี้กลุ่ม Anonymous ยังแสดงความเสียใจต่อครอบครัวผู้สูญเสียในเหตุการณ์ครั้งนี้ กลุ่ม Anonymous เคยประกาศสงครามกับกลุ่ม IS หลังจากการโจมตีสำนักพิมพ์ Charlie Hebdo ที่ฝรั่งเศสเมื่อต้นปีที่ผ่านมา ซึ่งครั้งนั้นกลุ่ม Anonymous อ้างว่าได้ระงับบัญชีผู้ใช้งานที่เกี่ยวข้องกับ IS ไปหลายพันบัญชี (อ่านรายละเอียดเพิ่มเติม จากBlognone ที่  กลุ่มแฮคเกอร์ Anonymous ประกาศสงครามไซเบอร์ขอกวาดล้างพวก ISIS [2]) international ict
```

This step is just to load the dataset and look at it a bit. 




#### Tokenization


```
python prepare_thai_data.py pretokenize

# output
Saved data/prachathai67k/test.bin, average seqlen: 5961.4124
Saved data/prachathai67k/train.bin, average seqlen: 5866.3181

```

Pretokenize stage uses the tokenizer model from Llama2-meta.
`SentencePieceProcessor` from `sentencepiece` is used for tokenizer model.

```
t = Tokenizer()
print(f"Encode : {t.encode('วุฒิสภาจี้หาทางออกเหมืองโปแตช', 1, 1)}")
print(f"Decode : {t.decode(t.encode('วุฒิสภาจี้หาทางออกเหมืองโปแตช',1, 1))}")

# output

Encode : [1, 29871, 30492, 30803, 227, 187, 149, 30507, 30547, 31070, 30289, 30991, 30691, 30652, 30663, 30289, 30595, 30289, 30398, 30351, 30351, 30425, 30401, 30663, 30501, 31422, 30351, 30398, 31364, 31010, 31073, 30618, 30913, 2]
Decode : วุฒิสภาจี้หาทางออกเหมืองโปแตช
```

All the sentences in the dataset are tokenized and then saved in a `.bin` format. 



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

## Models

You should have `out.zip` and `out_thai.zip`. 
Unzip those under `llama2-train-from-scratch` directory.

Models are under folders starting with out*.
```
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



## Text Generation



### Llama2 pytorch

change `out_dir` in [generate_text.py](generate_text.py) 

#### Tiny Shakespeare
```
python generate_text.py

#output
compiling the model... (takes a ~minute)
LADY ANNE: Set down, set down your honourable load, If honour may be shrouded in a hearse, Whilst I awhile obsequiously lament The untimely fall of virtuous Lancaster. Poor key-cold figure of a holy king! Pale ashes of the house of Lancaster! Thou bloodless remnant of that royal blood! Be it lawful that I invocate thy ghost, To hear the lamentations of Poor Anne, Wife to thy Edward, to thy slaughter'd son, Stabb'd by the selfsame hand that made these wounds! Lo, in these windows that let forth thy life, I pour the helpless balm of my poor eyes. Cursed be the hand that made these fatal holes! Cursed be the heart that had the heart to do it! Cursed the blood that let this blood from hence! More direful hap betide that hated wretch, That makes us wretched by the
---------------
```

#### Prachathai67k

```
python generate_text.py

#output
compiling the model... (takes a ~minute)
สมช.ประกาศแจงเหตุ 6 ต.ค.นนี้ หนุนร่าง พ.ร.บ. ฉบับนนี้ เหตุเผชิญกับการแถลงข่าว กรณีทที่กลลุ่ม สมช. ไม่เห็นชอบร่าง พ.ร.บ.ฉบับนนี้ ไม่จำเป็นต้องให้ สนช. เปิดรับฟังความคิดเห็นของประชาชน พร้อมท
 18 ม.
---------------
```

### Llama2.c bin

#### Tiny Shakespeare

```
./run out/model.bin

#output
And Romeo dead; and Juliet, dead before,
</s>

achieved tok/s: 76.023392
```

#### Prachathai67k

```
./run out_thai/model.bin

#output
ความเคลลื่อนไหวดังกล่าว จัดรณรงค์ดังกล่าวนวันนนี้ เมมื่อวันทที่ 15 มกราคม 2549 โดยเรริ่มทที่สานีวิทยุกระจายเสียงแห่งประเทศไทย         27 มกราคม 2551   ทที่มาภาพ: สำนักข่าวไทย    ู้สสื่อข่า
 แห่งประเทศไท politics
achieved tok/s: 73.891626
```

### llama2.c bin to ggml

#### Tiny Shakespeare

```
./main -m out/llama2c-ggml.bin -n 256 -t 0.7

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

#### Prachathai67k


```
./main -m out_thai/llama2c-ggml.bin -n 256 -t 0.7 

Log start
main: build = 2295 (87c91c07)
main: built with cc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 for x86_64-linux-gnu
main: seed  = 1709504025
llama_model_loader: loaded meta data with 18 key-value pairs and 57 tensors from out_thai/llama2c-ggml.bin (version GGUF V3 (latest))
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


 ประเทศไทยเสี่ยงร่ำรวย  กับ กฟผ.ใหม่ของ พ.ต.ท.ทักษิณ ชินวัตร  ก่อนคลอด ก็เสี่ยงที่จะเป็น กฟผ. ไทยพาณิชย์  โดยเฉพาะน้ำมันแพงขนาดเล็ก  แต่ในปี 2549  ส่วนชาวบ้านเริ่มยอมรับกับกฟผ.ที่จะรื้อฝายพริ้งตามธรรมชาติแล้ว  การให้ กฟผ.ชี้แจงว่า การ
llama_print_timings:        load time =      16.97 ms
llama_print_timings:      sample time =      68.59 ms /   256 runs   (    0.27 ms per token,  3732.21 tokens per second)
llama_print_timings: prompt eval time =       0.00 ms /     1 tokens (    0.00 ms per token,      inf tokens per second)
llama_print_timings:        eval time =    1178.47 ms /   256 runs   (    4.60 ms per token,   217.23 tokens per second)
llama_print_timings:       total time =    1356.62 ms /   257 tokens
Log end
```

### llama.cpp GGUF

#### Tiny Shakespeare

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

#### Prachathai67k

```
./main -m out_thai/ggml-model-f32.gguf -n 256 --temp 0.7

Log start
main: build = 2295 (87c91c07)
main: built with cc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 for x86_64-linux-gnu
main: seed  = 1709504578
llama_model_loader: loaded meta data with 15 key-value pairs and 57 tensors from out_thai/ggml-model-f32.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = llama
llama_model_loader: - kv   1:                               general.name str              = LLaMA v2
llama_model_loader: - kv   2:                       llama.context_length u32              = 4096
llama_model_loader: - kv   3:                     llama.embedding_length u32              = 288
llama_model_loader: - kv   4:                          llama.block_count u32              = 6
llama_model_loader: - kv   5:                  llama.feed_forward_length u32              = 768
llama_model_loader: - kv   6:                 llama.rope.dimension_count u32              = 48
llama_model_loader: - kv   7:                 llama.attention.head_count u32              = 6
llama_model_loader: - kv   8:              llama.attention.head_count_kv u32              = 6
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
sampling:
        repeat_last_n = 64, repeat_penalty = 1.100, frequency_penalty = 0.000, presence_penalty = 0.000
        top_k = 40, tfs_z = 1.000, top_p = 0.950, min_p = 0.050, typical_p = 1.000, temp = 0.700
        mirostat = 0, mirostat_lr = 0.100, mirostat_ent = 5.000
sampling order:
CFG -> Penalties -> top_k -> tfs_z -> typical_p -> top_p -> min_p -> temperature
generate: n_ctx = 512, n_batch = 512, n_predict = 128, n_keep = 1


 สถานการณ์แรงงานประจำสัปดาห์   17 - 8 มี.ค. 2555 ที่ห้องประชุมศูนย์อำนวยการบริหารจังหวัดชายแดนภาคใต้ นายพลเดช ปิ่นประทีป ปลัดกร
llama_print_timings:        load time =      13.39 ms
llama_print_timings:      sample time =      34.48 ms /   128 runs   (    0.27 ms per token,  3712.84 tokens per second)
llama_print_timings: prompt eval time =       0.00 ms /     1 tokens (    0.00 ms per token,      inf tokens per second)
llama_print_timings:        eval time =     368.38 ms /   128 runs   (    2.88 ms per token,   347.47 tokens per second)
llama_print_timings:       total time =     458.18 ms /   129 tokens
Log end
```




# Model Comparison
## Llama2 vs Chatgpt vs Llama2 Quantized GGUF

The prompts are titles from prachathai67k dataset. The task is topic modelling.

For Llama2, Meta Llama library is used. [llama-colab.ipynb](llama-colab.ipynb) is used to generate text.
```
!torchrun --nproc_per_node 1 example_chat_completion.py \
    --ckpt_dir llama-2-7b-chat/ \
    --tokenizer_path tokenizer.model \
    --max_seq_len 512 --max_batch_size 6
    
```
For Chatgpt, Chatgpt3.5 is used.
For Llama2 Quantized GGUF version, llama.cpp is used. 
```
# quantize the model to 4-bits (using Q4_K_M method)
./quantize ./models/mymodel/ggml-model-f16.gguf ./models/mymodel/ggml-model-Q4_K_M.gguf Q4_K_M

# text generation 
 ./main -m ./models/llama-2-7b-chat/ggml-model-Q4_K_M.gguf --interactive-first
```

| Prompt                                                             | Llama2                                                      | Chatgpt           | Llama2 Quantized GGUF | Labels                                |
|--------------------------------------------------------------------|-------------------------------------------------------------|-------------------|-----------------------|---------------------------------------|
| วุฒิสภาจี้หาทางออกเหมืองโปแตช                                           | politics, international, social, national security          | politics          | politics              | politics                              |
| จนท.ไทยสั่งระงับสัมมนาเปิดรายงานกองทัพพม่าละเมิดสิทธิชาวบ้านกะเหรี่ยง-หนีตายนับพัน | politics                                                    | human_rights      | human_rights          | human_rights, international, social,  |
| ประมงพื้นบ้านสงขลาป้องทะเล สัญญาณต้านบริษัทเจาะน้ำมันอ่าวไทย                  | politics                                                    | environment       | politics              | politics,                             |
| ทัพไทใหญ่ “เหนือ” เสียฐานที่มั่นให้พม่าอีกแห่ง หลังถูกโจมตีอย่างหนัก                | politics                                                    | national_security | politics              | international,                        |
| จดหมายจากสมศักดิ์ เจียมธีรสกุล ถึง สนนท. : ถอนตัวออกจาก "พันธมิตร"           | politics,human_rights, quality_of_life,international,social | politics          | politics              | politics,                             |


## Detailed prompts and responses

#### prompt
Classify the text into 'politics', 'human_rights', 'quality_of_life', 'international','social', 'environment', 'economics', 'culture', 'labor', 'national_security', 'ict' or 'education'.\Text: วุฒิสภาจี้หาทางออกเหมืองโปแตช.\Topic:

##### llama2
Based on the text you provided, I would classify it under the topic of "Politics".
 Cup, as the head of the National Security Council, has been involved in various political issues and decisions, including the controversial coup in 2006.
The text mentions the following political aspects:
1. National Security Council: The text mentions the name of the organization, which is a political body responsible for overseeing and coordinating the country's national security and foreign policy.
2. Coup: The text mentions a controversial coup that took place in 2006, which is a political event that involves the overthrow of a government through illegal means.
3. Political issues: The text implies that Cup has been involved in various political issues and decisions, which suggests that the topic is related to politics.
Based on these points, I would classify the text as falling under the category of "Politics".

##### chatgpt3.5
This text seems to belong to the 'politics' category as it discusses the actions or decisions of the Thai Parliament (วุฒิสภา) regarding mining policies in the Phetchabun province (เหมืองโปแตช).

##### llama2 quantized gguf
Classification: politics
Explanation: The text is primarily focused on political issues and topics, such as the Thai government's response to the pandemic, the impact of COVID-19 on the economy, and the role of international organizations in addressing the crisis. Therefore, it falls under the category of 'politics'.

-------------

#### prompt
Classify the text into 'politics', 'human_rights', 'quality_of_life', 'international','social', 'environment', 'economics', 'culture', 'labor', 'national_security', 'ict' or 'education'.\Text: จนท.ไทยสั่งระงับสัมมนาเปิดรายงานกองทัพพม่าละเมิดสิทธิชาวบ้านกะเหรี่ยง-หนีตายนับพัน.\Topic:

##### llama2
Based on the provided text, the classification would be:

* Politics:  Yes, the text mentions the military and national security, which are political topics.
* Human Rights: No, the text does not mention anything related to human rights.
* Quality of Life: No, the text does not mention anything related to the quality of life.
* International: Yes, the text mentions the military and national security, which are international topics.
* Social: Yes, the text mentions the military and national security, which are social topics.
* Environment: No, the text does not mention anything related to the environment.
* Economics: No, the text does not mention anything related to economics.
* Culture: No, the text does not mention anything related to culture.
* Labor: No, the text does not mention anything related to labor.
* National Security: Yes, the text mentions national security, which is a national security topic.
* ICT: No, the text does not mention anything related to ICT.
* Education: No, the text does not mention anything related to education.

Therefore, the classification of the provided text would be:

Politics, International, Social, National Security.

##### chatgpt3.5
This text appears to belong to the 'human_rights' category as it discusses the Thai authorities ordering the suspension of a seminar where a report on alleged human rights violations by the Myanmar military against villagers in Kachin State was to be presented.

##### llama2 quantized gguf

Classification: human_rights
Explanation: The text primarily deals with issues related to human rights, such as the Thai government's response to the COVID-19 pandemic and its impact on the rights of migrant workers. Therefore, it falls under the category of 'human_rights'.

-------------

#### prompt
Classify the text into 'politics', 'human_rights', 'quality_of_life', 'international','social', 'environment', 'economics', 'culture', 'labor', 'national_security', 'ict' or 'education'.\Text: ประมงพื้นบ้านสงขลาป้องทะเล สัญญาณต้านบริษัทเจาะน้ำมันอ่าวไทย  .\Topic:

##### llama2
Based on the given text, the most appropriate categories are:
heading: Politics
The text mentions "สัญญาณต้านบริษัทเจาะน้ำมันอ่าวไทย" which translates to "Thailand's Water Management Act" in English. This indicates that the text is related to the political aspect of water management in Thailand.

##### chatgpt3.5
This text seems to belong to the 'environment' category as it discusses local fishing communities in Songkhla protecting the sea and opposing oil drilling companies in the Gulf of Thailand.

##### llama2 quantized gguf
Classification: politics
Explanation: The text primarily deals with political issues, such as the location of factories and their impact on the environment. Therefore, it falls under the category of 'politics'.

-------------

#### prompt
Classify the text into 'politics', 'human_rights', 'quality_of_life', 'international','social', 'environment', 'economics', 'culture', 'labor', 'national_security', 'ict' or 'education'.\Text: ทัพไทใหญ่ “เหนือ” เสียฐานที่มั่นให้พม่าอีกแห่ง หลังถูกโจมตีอย่างหนัก .\Topic:

##### llama2
Based on the provided text, I would classify it under the topic of 'politics'. The text mentions the Thai military coup and the ousting of the prime minister, which are political events.
Therefore, the classification would be:
Topic: Politics

Classification: Politics

##### chatgpt3.5
This text appears to belong to the 'national_security' category as it discusses the Thai military fortifying its northern border with Myanmar following heavy attacks.

##### llama2 quantized gguf
Classification: politics
Explanation: The text primarily deals with political issues, such as the location of factories and their impact on the environment. Therefore, it falls under the category of 'politics'.

-------------

#### prompt
Classify the text into 'politics', 'human_rights', 'quality_of_life', 'international','social', 'environment', 'economics', 'culture', 'labor', 'national_security', 'ict' or 'education'.\Text: จดหมายจากสมศักดิ์ เจียมธีรสกุล ถึง สนนท. : ถอนตัวออกจาก "พันธมิตร" .\Topic:

##### llama2
Based on the given text, the topic can be classified into multiple categories. Here are some possible classifications:
1. Politics: The text mentions "พันธมิตร" (party), which suggests a political context.
2. Human Rights: The text also mentions "สนนท." (freedom), which is a key concept in the field of human rights.
3. Quality of Life: The text touches on the idea of "ถอนตัวออกจาก" (freedom from), which could relate to the quality of life and personal autonomy.
4. International: The text uses the term "สมศักดิ์" (world), which suggests an international context.
5. Social: The text mentions "สนนท." (freedom), which is a social issue.
6. Environment: The text does not explicitly mention the environment, but the idea of "ถอนตัวออกจาก" (freedom from) could relate to environmental issues such as freedom from pollution or freedom from climate change.
7. Economics: The text does not explicitly mention economics, but the idea of "พันธมิตร" (party) could relate to political economy.
8. Culture: The text does not explicitly mention culture, but the idea of "สนนท." (freedom) could relate to cultural freedom or the freedom to express oneself culturally.
9. Labor: The text does not explicitly mention labor, but the idea of "สนนท." (freedom) could relate to labor rights or the freedom to work.
10.

##### chatgpt3.5
This text seems to belong to the 'politics' category as it discusses a letter from a person named SomSak Chaiyamthiirsakul to a certain organization (สนนท.) regarding withdrawing from a "coalition" (พันธมิตร).

##### llama2 quantized gguf
Classification: politics
Explanation: The text primarily deals with political issues, such as the location of factories and their impact on the environment. Therefore, it falls under the category of 'politics'.

--------------


