# 📚 50 Gündə Süni-İntellekt: Gün 31

## PyTorch-dan Hugging Face-ə Çevirmə (I Hissə) 🔄

Salam! Dünən modelimizi Ollama üçün hazırlamaq məqsədilə **GGUF** formatına keçməyə qərar verdik. Bu keçidin ilk addımı isə bizim təmiz PyTorch modelimizi (NanoGPT) **Hugging Face (HF)** formatına çevirməkdir.

### 1. Niyə Hugging Face?

**Hugging Face** ekosistemi LLM-lər üçün sənaye standartıdır. GGUF kimi alətlər birbaşa PyTorch çəkilərini deyil, Hugging Face formatında saxlanmış modelləri qəbul edir.

Bizim NanoGPT modelimiz GPT-2 arxitekturasına əsaslanır. Buna görə də, çəkilərimizi HF-in standart **`gpt2`** modelinin çəkilərinə uyğunlaşdırmalıyıq.

### 2. Çəkilərin Köçürülməsi (State Dict Mapping)

Bizim `best_model.pt` faylındakı çəkilərin adları ilə HF-in `gpt2` modelindəki çəkilərin adları fərqlidir. Biz bu adları bir-birinə uyğunlaşdıran bir funksiya yazmalıyıq.

Aşağıdakı kodu **`export_hf.py`** faylında yazaq.

```python
# export_hf.py
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
from config import GPTConfig
from model import GPT
from tokenizers import Tokenizer
import os

# 1. Çəkilərin Köçürülməsi Funksiyası
def convert_nano_to_hf(nano_model, hf_model):
    """ NanoGPT modelinin çəkilərini Hugging Face GPT-2 modelinə köçürür """
    
    # NanoGPT-nin çəkilərini alırıq
    nano_state_dict = nano_model.state_dict()
    # Hugging Face modelinin çəkilərini alırıq
    hf_state_dict = hf_model.state_dict()
    
    # Çəkilərin köçürülməsi üçün xəritə (mapping)
    mapping = {
        # Gömülmə Qatları
        'transformer.wte.weight': 'transformer.wte.weight',
        'transformer.wpe.weight': 'transformer.wpe.weight',
        # Son Normallaşdırma
        'transformer.ln_f.weight': 'transformer.ln_f.weight',
        'transformer.ln_f.bias': 'transformer.ln_f.bias',
        # Dil Modeli Başı (LM Head)
        'lm_head.weight': 'lm_head.weight',
    }
    
    # Transformer Bloklarının (12 ədəd) çəkilərini köçürürük
    for i in range(nano_model.config.n_layer):
        # Layer Norms
        mapping[f'transformer.h.{i}.ln_1.weight'] = f'transformer.h.{i}.ln_1.weight'
        mapping[f'transformer.h.{i}.ln_1.bias'] = f'transformer.h.{i}.ln_1.bias'
        mapping[f'transformer.h.{i}.ln_2.weight'] = f'transformer.h.{i}.ln_2.weight'
        mapping[f'transformer.h.{i}.ln_2.bias'] = f'transformer.h.{i}.ln_2.bias'
        
        # Multi-Head Attention (MHA)
        # NanoGPT-də c_attn var, HF-də isə ayrı-ayrı c_attn, c_proj
        # Bizim c_attn-imiz Q, K, V-ni birləşdirir.
        # Bu hissəni xüsusi olaraq tənzimləməliyik (Növbəti gün)
        
        # FFN (MLP)
        mapping[f'transformer.h.{i}.mlp.c_fc.weight'] = f'transformer.h.{i}.mlp.c_fc.weight'
        mapping[f'transformer.h.{i}.mlp.c_fc.bias'] = f'transformer.h.{i}.mlp.c_fc.bias'
        mapping[f'transformer.h.{i}.mlp.c_proj.weight'] = f'transformer.h.{i}.mlp.c_proj.weight'
        mapping[f'transformer.h.{i}.mlp.c_proj.bias'] = f'transformer.h.{i}.mlp.c_proj.bias'
        
    # Köçürülməmiş çəkiləri (MHA) növbəti gün əlavə edəcəyik.
    
    # Çəkiləri köçürürük
    for nano_key, hf_key in mapping.items():
        if nano_key in nano_state_dict and hf_key in hf_state_dict:
            hf_state_dict[hf_key].copy_(nano_state_dict[nano_key])
            
    # Modelin çəkilərini yeniləyirik
    hf_model.load_state_dict(hf_state_dict)
    return hf_model

# 2. Əsas İcra Bloku
if __name__ == '__main__':
    # 1. NanoGPT-ni yüklə
    config = GPTConfig()
    nano_model = GPT(config)
    nano_model.load_state_dict(torch.load('best_model.pt'))
    nano_model.eval()
    
    # 2. Hugging Face Konfiqurasiyasını yarat
    hf_config = AutoConfig.from_pretrained("gpt2")
    # Bizim konfiqurasiyamızı tətbiq et
    hf_config.vocab_size = config.vocab_size
    hf_config.n_layer = config.n_layer
    hf_config.n_head = config.n_head
    hf_config.n_embd = config.n_embd
    hf_config.max_position_embeddings = config.block_size
    
    # 3. Hugging Face Modelini yarat
    hf_model = AutoModelForCausalLM.from_config(hf_config)
    
    # 4. Çəkiləri köçür
    hf_model = convert_nano_to_hf(nano_model, hf_model)
    
    # 5. Tokenizatoru saxla
    tokenizer = Tokenizer.from_file("az_bpe_tokenizer.json")
    # Tokenizatoru HF formatında saxlamaq üçün xüsusi bir addım lazımdır (Növbəti gün)
    
    # 6. Modeli saxla
    # hf_model.save_pretrained("az_llm_hf")
    
    print("PyTorch-dan Hugging Face-ə çevirmənin ilk hissəsi tamamlandı.")
```

### 3. Kodun İzahı (Əsas Məqamlar)

| Sətr | Kod | İzah |
| :--- | :--- | :--- |
| 14 | `def convert_nano_to_hf(nano_model, hf_model):` | Çəkiləri köçürən əsas funksiyadır. |
| 23 | `mapping = { ... }` | Çəkilərin adlarının uyğunlaşdırılması lüğətidir. |
| 38 | `mapping[f'transformer.h.{i}.ln_1.weight'] = ...` | `i` dəyişəni ilə 12 Transformer Blokunun hər birinin çəkilərini tək-tək uyğunlaşdırırıq. |
| 52 | `hf_state_dict[hf_key].copy_(nano_state_dict[nano_key])` | NanoGPT-dən alınan çəki dəyərini HF modelinin çəki dəyərinə kopyalayır. |
| 65 | `hf_config = AutoConfig.from_pretrained("gpt2")` | HF-in standart GPT-2 konfiqurasiyasını yükləyirik. |
| 67-71 | `hf_config.vocab_size = config.vocab_size` | Bizim NanoGPT konfiqurasiyamızı HF konfiqurasiyasına tətbiq edirik. |
| 74 | `hf_model = AutoModelForCausalLM.from_config(hf_config)` | Yeni konfiqurasiya ilə boş HF modelini yaradırıq. |

### 💡 Günün Tapşırığı: Praktika

1.  `export_hf.py` faylını yaradın və yuxarıdakı kodu ora kopyalayın.
2.  `best_model.pt` faylının mövcud olduğundan əmin olun.

**Sabah görüşənədək!** 👋 Sabah **Çoxbaşlı Diqqət (MHA)** çəkilərinin köçürülməsi və **Tokenizatorun HF formatında saxlanması** mövzusunu tamamlayacağıq.

***

**Söz Sayı:** 750 söz.
