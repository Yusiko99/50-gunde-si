# Gün 33: GGUF Formatına Çevirmə (Kvantlaşdırma) 📦

## 33.1. Kvantlaşdırma Prosesinin Məntiqi

Gün 32-də modelimizi Hugging Face (HF) formatına çevirdik. İndi bu HF modelini **GGUF (GPT-GEneration.cpp Unified Format)** formatına çevirməliyik.

**Məntiq:** GGUF formatı, modelin çəkilərini **Int4** kimi aşağı dəqiqliyə çevirərək modelin ölçüsünü 8 dəfə azaldır. Bu, modelin məhdud VRAM-lı GPU-larda (məsələn, 4GB RTX 2050) və ya CPU-da sürətli işləməsini təmin edir.

Kvantlaşdırma prosesi adətən **Llama.cpp** layihəsinin alətləri ilə həyata keçirilir. Bu proses iki əsas addımdan ibarətdir:

1.  **Xam Çevrilmə:** HF modelini xam FP32 GGUF formatına çevirmək.
2.  **Kvantlaşdırma:** Xam GGUF-u Int4 (Q4_0) formatına çevirmək.

## 33.2. Praktika: Kvantlaşdırma Skripti (Simulyasiya)

Kvantlaşdırma prosesi bir neçə terminal əmri tələb etdiyi üçün, biz bu prosesi simulyasiya edən və əsas məntiqi izah edən bir skript təqdim edirik.

**`quantize_to_gguf.py`**

```python
import os
import subprocess

HF_MODEL_PATH = "az_llm_hf"
OUTPUT_DIR = "az_llm_gguf"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Xam GGUF Faylının Adı (FP32)
RAW_GGUF_FILE = os.path.join(OUTPUT_DIR, "az_llm_f32.gguf")

# 2. Kvantlaşdırılmış GGUF Faylının Adı (Q4_0)
FINAL_GGUF_FILE = os.path.join(OUTPUT_DIR, "az_llm_100m_q4_0.gguf")
QUANTIZATION_TYPE = "Q4_0" # 4-bit kvantlaşdırma

def simulate_quantization():
    """GGUF çevrilmə və kvantlaşdırma prosesini simulyasiya edir."""
    
    print(f"1. Hugging Face modelinin xam GGUF-a çevrilməsi...")
    # Realda: llama.cpp/convert.py az_llm_hf --outtype f32 --outfile az_llm_gguf/az_llm_f32.gguf
    
    # Simulyasiya: Xam GGUF faylını yaratmaq
    with open(RAW_GGUF_FILE, 'w') as f:
        f.write("Bu fayl modelin FP32 çəkilərini ehtiva edir.")
        
    print(f"Xam GGUF faylı yaradıldı: {RAW_GGUF_FILE}")
    
    print(f"\n2. Kvantlaşdırma ({QUANTIZATION_TYPE}) prosesi...")
    # Realda: llama.cpp/quantize az_llm_gguf/az_llm_f32.gguf az_llm_gguf/az_llm_q4_0.gguf Q4_0
    
    # Simulyasiya: Kvantlaşdırılmış GGUF faylını yaratmaq
    with open(FINAL_GGUF_FILE, 'w') as f:
        f.write(f"Bu fayl {QUANTIZATION_TYPE} kvantlaşdırılmış GGUF modelini ehtiva edir.")
        
    print(f"Kvantlaşdırma tamamlandı. Yekun GGUF faylı: {FINAL_GGUF_FILE}")
    print(f"Modelin ölçüsü təxminən 50-70MB olacaq.")

if __name__ == "__main__":
    if not os.path.exists(HF_MODEL_PATH):
        print("Xəta: Hugging Face model qovluğu tapılmadı. Zəhmət olmasa Gün 32-ni tamamlayın.")
    else:
        simulate_quantization()
```

## 33.3. Kvantlaşdırmanın Məntiqi İzahı

| Addım | Məntiqi Əsas |
| :--- | :--- |
| **Xam Çevrilmə** | HF modelinin çəkiləri (məsələn, `pytorch_model.bin`) Llama.cpp-nin daxili formatına (GGUF) köçürülür. Bu mərhələdə dəqiqlik (FP32) saxlanılır. |
| **Kvantlaşdırma** | **Kritik Addım.** Bu alət, FP32 çəkilərini oxuyur və onları **4-bitlik (Int4)** tam ədədlərə çevirir. Bu çevrilmə zamanı modelin dəqiqliyində minimal itki ilə yaddaş tələbi kəskin şəkildə azalır. |
| **Q4_0** | **Q**uantization **4**-bit **0**-cu versiya deməkdir. Bu, ən çox istifadə olunan və ən yüngül kvantlaşdırma növüdür. |

**Nəticə:** Bu prosesin sonunda əldə edilən **`az_llm_100m_q4_0.gguf`** faylı, modelin bütün biliklərini 50-70MB həcmində ehtiva edir və Ollama-da istifadə üçün hazırdır.
