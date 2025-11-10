# Gün 30: Modelin Yüngülləşdirilməsi (Quantization) ⚖️

## 30.1. Quantization-ın Zəruriliyi

**Quantization (Kvantlaşdırma)** modelin çəkilərini daha az bit dəqiqliyinə (məsələn, 32-bitdən 4-bitə) çevirmə prosesidir.

**Məntiq:** LLM-lər adətən 32-bitlik (FP32) dəqiqlikdə təlim keçir. Lakin proqnozlaşdırma (Inference) zamanı bu yüksək dəqiqliyə ehtiyac qalmır. Kvantlaşdırma modelin ölçüsünü kəskin şəkildə azaldır və proqnozlaşdırma sürətini artırır.

| Dəqiqlik | Yaddaş Tələbi (100M Parametr) | Sürət |
| :--- | :--- | :--- |
| **FP32** | 400 MB | Standart |
| **FP16** | 200 MB | Sürətli |
| **Int8** | 100 MB | Çox Sürətli |
| **Int4** | **50 MB** | Ən Sürətli və Yüngül |

**Kritik Məntiq:** Məhdud resurslu cihazlarda (məsələn, 4GB VRAM-lı RTX 2050) modelin yaddaşda saxlanması və işlədilməsi üçün Int4 kvantlaşdırması həyati əhəmiyyət daşıyır.

## 30.2. GGUF Formatına Giriş

Biz modelimizi **GGUF (GPT-GEneration.cpp Unified Format)** formatına çevirəcəyik.

*   **GGUF** **llama.cpp** layihəsi tərəfindən yaradılmışdır.
*   Bu format, kvantlaşdırılmış modellərin CPU və ya məhdud VRAM-lı GPU-larda (məsələn, RTX 2050) effektiv şəkildə işlədilməsi üçün optimallaşdırılmışdır.
*   **Ollama** (Gün 34-də öyrəniləcək) məhz GGUF formatını dəstəkləyir.

**GGUF-un Üstünlükləri:**

1.  **Hər Şey Bir Faylda:** Modelin arxitekturası, çəkiləri və tokenizatoru tək bir `.gguf` faylında saxlanılır.
2.  **Müxtəlif Kvantlaşdırma Səviyyələri:** Int4, Int5, Int8 kimi müxtəlif kvantlaşdırma səviyyələrini dəstəkləyir.

## 30.3. Kvantlaşdırma Prosesinin Mərhələləri

Kvantlaşdırma prosesi bir neçə ardıcıl addımdan ibarətdir:

1.  **PyTorch-dan Hugging Face-ə Çevirmə:** Bizim sıfırdan qurduğumuz PyTorch modelini sənaye standartı olan Hugging Face (HF) formatına çevirmək.
2.  **HF-dən Llama.cpp-yə Çevirmə:** HF modelini Llama.cpp-nin başa düşdüyü xam formata çevirmək.
3.  **Kvantlaşdırma:** Llama.cpp-nin alətləri ilə xam modeli Int4 GGUF formatına çevirmək.

**Məntiq:** Bu çevrilmə zənciri, modelin PyTorch-da öyrəndiyi bilikləri, ən yüngül və ən sürətli proqnozlaşdırma mühitinə (Ollama/Llama.cpp) köçürməyə imkan verir.

## 30.4. Günün Tapşırığı: Hazırlıq

Növbəti mərhələlər üçün lazım olan əsas kitabxanaları quraşdırmaq:

```bash
# Hugging Face Transformers kitabxanası (Çevirmə üçün)
pip install transformers

# Llama.cpp ilə işləmək üçün Python interfeysi
pip install llama-cpp-python
```
