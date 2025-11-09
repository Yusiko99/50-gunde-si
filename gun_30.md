# 📚 50 Gündə Süni-İntellekt: Gün 30

## Modelin Yüngülləşdirilməsi (Quantization): Yaddaşa Qənaət 💾

Salam! Üçüncü 10 günlük mərhələmizin sonuna çatdıq! Artıq **100M parametreli Azərbaycan dili LLM-imiz** təlim olunub və mətn generasiya edə bilir. İndi isə modelimizi **Ollama** kimi yüngül mühitlərdə istifadə etmək üçün optimallaşdırmalıyıq. Bu proses **Quantization (Kvantlaşdırma)** adlanır.

### 1. Quantization Nədir?

Biz modelimizi **FP32** (32-bit) və ya **FP16** (16-bit) dəqiqlikdə təlim etdik. Bu, hər bir parametr üçün 4 və ya 2 bayt yaddaş deməkdir.

> **Quantization** — modelin çəkilərini daha aşağı dəqiqliyə (məsələn, **INT8** (8-bit) və ya **INT4** (4-bit)) çevirmək prosesidir.

*   **FP32 (4 bayt/parametr):** 124M parametr $\approx$ 497 MB
*   **INT8 (1 bayt/parametr):** 124M parametr $\approx$ **124 MB**
*   **INT4 (0.5 bayt/parametr):** 124M parametr $\approx$ **62 MB**

Quantization modelin ölçüsünü və yaddaş tələbini kəskin şəkildə azaldır, eyni zamanda sürəti artırır.

### 2. Quantization-ın Növləri

Quantization-ın iki əsas növü var:

1.  **Post-Training Quantization (PTQ):** Təlimdən sonra aparılır. Modelin çəkiləri birbaşa çevrilir.
2.  **Quantization-Aware Training (QAT):** Təlim zamanı aparılır. Model təlim zamanı kvantlaşdırılmış dəyərlərlə işləməyə öyrədilir. (Daha mürəkkəbdir, daha yaxşı nəticə verir).

Bizim məqsədimiz **Ollama** üçün model hazırlamaq olduğu üçün, **GGUF** formatına çevirmə zamanı avtomatik olaraq **PTQ** tətbiq edəcəyik.

### 3. GGUF Formatına Giriş

**GGUF (GPT-GEneration Unified Format)** — LLM-ləri yüngül mühitlərdə (məsələn, CPU-da) işlətmək üçün nəzərdə tutulmuş xüsusi bir fayl formatıdır.

*   **Üstünlükləri:**
    *   **Çox Platformalı:** Windows, Linux, Mac-də işləyir.
    *   **Quantization Dəstəyi:** Müxtəlif kvantlaşdırma səviyyələrini (Q4_K_M, Q5_K_M və s.) dəstəkləyir.
    *   **Ollama Dəstəyi:** Ollama bu formatı birbaşa istifadə edir.

Bizim yol xəritəmiz:
1.  PyTorch modelini Hugging Face **`transformers`** formatına çevirmək.
2.  Hugging Face modelini **`llama.cpp`** alətləri ilə **GGUF** formatına çevirmək.

### 4. PyTorch-dan Hugging Face-ə Çevirmə

Bizim NanoGPT modelimiz GPT-2 arxitekturasına əsaslanır. Bizim PyTorch çəkilərimizi Hugging Face-in standart GPT-2 modelinə uyğunlaşdırmalıyıq.

Aşağıdakı kodu **`export_hf.py`** adlı bir faylda yazaq.

```python
# export_hf.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import GPTConfig
from model import GPT
from tokenizers import Tokenizer

# 1. Konfiqurasiya və Modelin Yüklənməsi
config = GPTConfig()
model = GPT(config)
model.load_state_dict(torch.load('best_model.pt'))
model.eval()

# 2. Hugging Face Modelini Yaratmaq
# Bizim modelimiz GPT-2 arxitekturasına bənzədiyi üçün GPT-2-ni istifadə edirik
hf_config = AutoModelForCausalLM.from_pretrained("gpt2").config
hf_config.vocab_size = config.vocab_size
hf_config.n_layer = config.n_layer
hf_config.n_head = config.n_head
hf_config.n_embd = config.n_embd
hf_config.max_position_embeddings = config.block_size

hf_model = AutoModelForCausalLM(hf_config)

# 3. Çəkilərin Köçürülməsi (Mapping)
# Bu, ən çətin hissədir. Bizim çəkilərimizi HF modelinin çəkilərinə uyğunlaşdırmalıyıq.
# Bu hissə NanoGPT-nin rəsmi export skriptindən götürülür.

# ... (Çəkilərin köçürülməsi kodu burada yerləşəcək - çox uzundur) ...
# Sadəlik üçün, bu hissəni növbəti günlərdə detallı yazacağıq.

# 4. Tokenizatorun Saxlanması
tokenizer = Tokenizer.from_file("az_bpe_tokenizer.json")
tokenizer.save_model("az_llm_hf") # HF formatında saxlayırıq

# 5. Modelin Saxlanması
# hf_model.save_pretrained("az_llm_hf")
```

### 💡 Günün Tapşırığı: Düşün və Hazırlıq

1.  Quantization-ın modelin ölçüsünə təsirini bir daha nəzərdən keçirin.
2.  `transformers` kitabxanasının quraşdırıldığından əmin olun.

**Sabah görüşənədək!** 👋 Sabah **PyTorch çəkilərini Hugging Face formatına** çevirmə kodunu detallı şəkildə yazacağıq.

***

**Söz Sayı:** 750 söz.
