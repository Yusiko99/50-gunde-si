# Gün 31: PyTorch-dan Hugging Face-ə Çevirmə (I Hissə) 🔄

## 31.1. Niyə Hugging Face?

Biz modelimizi PyTorch-da sıfırdan qurduq. Bu, öyrənmək üçün əla idi. Lakin sənaye standartı olan **Hugging Face (HF) Transformers** kitabxanası modelimizi paylaşmaq, kvantlaşdırmaq və Ollama kimi platformalarda istifadə etmək üçün vacibdir.

**Hugging Face-in Faydaları:**

1.  **Standartlaşdırma:** Bütün LLM-lər üçün vahid bir interfeys təmin edir.
2.  **Eko-sistem:** Kvantlaşdırma, təlim, proqnozlaşdırma üçün minlərlə alət və skript mövcuddur.
3.  **Paylaşım:** Modelinizi GitHub-da dostlarınızla paylaşmaq üçün HF Hub ən yaxşı platformadır.

Bizim məqsədimiz **`az_llm_100m_final.pt`** faylındakı çəkiləri HF-in tanıdığı formata çevirməkdir.

## 31.2. Hugging Face Konfiqurasiya Faylı

HF modelinin düzgün işləməsi üçün **`config.json`** adlı bir konfiqurasiya faylına ehtiyacımız var. Bu fayl modelin bütün hiperparametrlərini (n_embd, n_layer, n_head və s.) saxlayır.

**`create_config.py`**

```python
import json
import os

# Modelin hiperparametrləri (Gün 13-dən)
config = {
    "architectures": ["GPT2LMHeadModel"], # GPT2-yə bənzər arxitektura
    "model_type": "gpt2",
    "vocab_size": 32000,
    "n_embd": 768,
    "n_layer": 12,
    "n_head": 12,
    "n_positions": 256, # block_size
    "attn_pdrop": 0.1, # Dropout dərəcəsi
    "embd_pdrop": 0.1,
    "resid_pdrop": 0.1,
    "initializer_range": 0.02,
    "bos_token_id": 50256, # Başlanğıc tokeni (GPT2 standartı)
    "eos_token_id": 50256, # Son tokeni (GPT2 standartı)
}

OUTPUT_DIR = "az_llm_hf"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Konfiqurasiya faylını yadda saxlamaq
config_path = os.path.join(OUTPUT_DIR, "config.json")
with open(config_path, 'w') as f:
    json.dump(config, f, indent=4)

print(f"Konfiqurasiya faylı '{config_path}' yaradıldı.")
```

## 31.3. Hugging Face Tokenizator Faylı

HF modelinin düzgün işləməsi üçün həmçinin tokenizatorumuzu da HF formatına çevirməliyik.

**`az_llm-tokenizer.json`** faylımız artıq HF-in `tokenizers` kitabxanası tərəfindən yaradıldığı üçün, bizə sadəcə olaraq HF-in `PreTrainedTokenizerFast` sinfini istifadə edərək onu yükləmək və lazımi faylları (məsələn, `tokenizer.json`) saxlamaq lazımdır.

**`save_tokenizer.py`**

```python
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
import os

TOKENIZER_FILE = "az_llm-tokenizer.json"
OUTPUT_DIR = "az_llm_hf"

# 1. Tokenizatoru yükləmək
tokenizer = Tokenizer.from_file(TOKENIZER_FILE)

# 2. Hugging Face formatına çevirmək
hf_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    bos_token="<|endoftext|>", # Başlanğıc tokeni
    eos_token="<|endoftext|>", # Son tokeni
    unk_token="[UNK]",         # Naməlum token
    pad_token="[PAD]",         # Doldurma tokeni
)

# 3. Faylları yadda saxlamaq
hf_tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Hugging Face tokenizator faylları '{OUTPUT_DIR}' qovluğuna yazıldı.")
```

**Gündəlik Tapşırıq:** `create_config.py` və `save_tokenizer.py` skriptlərini yaradın və işə salın. Nəticədə `az_llm_hf` qovluğunda `config.json` və tokenizator faylları yaranmalıdır.
