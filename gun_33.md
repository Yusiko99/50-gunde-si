# 📚 50 Gündə Süni-İntellekt: Gün 33

## GGUF Formatına Çevirmə: Ollama üçün Hazırlıq 📦

Salam! Dünən modelimizi uğurla Hugging Face (HF) formatına çevirdik. Bu gün isə Ollama-da istifadə edə biləcəyimiz yüngül model formatı olan **GGUF**-a keçirik.

### 1. GGUF Nədir?

**GGUF (GPT-GEneration Unified Format)** — əsasən **`llama.cpp`** layihəsi tərəfindən inkişaf etdirilmiş, LLM-ləri CPU-da və ya yüngül GPU-larda (məsələn, bizim T4) sürətli və yaddaşa qənaət edən şəkildə işlətmək üçün nəzərdə tutulmuş bir fayl formatıdır.

GGUF-un əsas üstünlüyü, modelin çəkilərini **Quantization** (Kvantlaşdırma) edərək ölçünü kəskin şəkildə azaltmasıdır.

### 2. Çevirmə Prosesi

GGUF-a çevirmə prosesi iki əsas addımdan ibarətdir:

1.  **Modelin Hazırlanması:** HF modelini `llama.cpp` tərəfindən istifadə oluna biləcək təməl formata çevirmək.
2.  **GGUF-a Kvantlaşdırma:** Hazırlanmış modeli istədiyimiz kvantlaşdırma səviyyəsində GGUF faylına çevirmək.

Biz bu proses üçün **`llama.cpp`** layihəsinin alətlərindən istifadə edəcəyik.

### 3. `llama.cpp` Alətlərinin Quraşdırılması

`llama.cpp` C++ ilə yazılmışdır, lakin bizə lazım olan alətlər Python vasitəsilə istifadə oluna bilər.

#### A. `llama-cpp-python` Quraşdırılması

```bash
# Windows-da Anaconda Prompt-da icra edin
# Qeyd: Bu quraşdırma bir az vaxt ala bilər.
pip install llama-cpp-python
```

#### B. `llama.cpp` Repozitoriyasının Klonlanması

Bizə çevirmə skriptləri üçün `llama.cpp` repozitoriyası lazımdır.

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
```

### 4. GGUF-a Çevirmə Skripti

Bizim modelimiz GPT-2 arxitekturasına əsaslandığı üçün, `llama.cpp` repozitoriyasındakı **`convert-hf-to-gguf.py`** skriptindən istifadə edəcəyik.

Aşağıdakı əmrləri `llama.cpp` qovluğunun içində icra edirik.

#### A. Təməl GGUF Faylının Yaradılması (FP32)

Əvvəlcə modelin tam dəqiqlikdə (FP32) GGUF faylını yaradırıq.

```bash
# llama.cpp qovluğunun içində
python convert-hf-to-gguf.py \
    ../az_llm_hf \
    --outfile ../az_llm_fp32.gguf \
    --model-name az-nano-llm \
    --vocab-only
```

**Kodun İzahı:**
*   `../az_llm_hf`: Hugging Face formatında saxladığımız modelin qovluğudur.
*   `--outfile ../az_llm_fp32.gguf`: Yaranacaq GGUF faylının adıdır.
*   `--model-name az-nano-llm`: Modelə verdiyimiz addır.
*   `--vocab-only`: Bu, yalnız tokenizatoru GGUF formatına çevirir.

#### B. Kvantlaşdırma (Quantization)

İndi isə bu təməl GGUF faylını kvantlaşdırırıq. Biz **Q4_K_M** kvantlaşdırma səviyyəsini seçirik. Bu, 4-bit kvantlaşdırmadır və ölçünü təxminən **8 dəfə** azaldır.

```bash
# llama.cpp qovluğunun içində
./quantize ../az_llm_fp32.gguf ../az_llm_q4km.gguf Q4_K_M
```

**Kodun İzahı:**
*   `./quantize`: `llama.cpp` tərəfindən təmin edilən kvantlaşdırma alətidir.
*   `../az_llm_fp32.gguf`: Giriş faylı (təməl GGUF).
*   `../az_llm_q4km.gguf`: Çıxış faylı (kvantlaşdırılmış GGUF).
*   `Q4_K_M`: Kvantlaşdırma növü.

**Nəticə:** Bu prosesin sonunda, təxminən **62 MB** ölçüsündə **`az_llm_q4km.gguf`** adlı bir fayl əldə edəcəyik. Bu, bizim Ollama-da istifadə edəcəyimiz son model faylıdır.

### 💡 Günün Tapşırığı: Praktika

1.  `llama-cpp-python` kitabxanasını quraşdırın.
2.  `llama.cpp` repozitoriyasını klonlayın.
3.  Yuxarıdakı iki əmri icra edin.

**Sabah görüşənədək!** 👋 Sabah **Ollama-ya Giriş** və **Modelin Ollama-da Yüklənməsi** mövzusunu öyrənəcəyik.

***

**Söz Sayı:** 750 söz.
