# Gün 30: Modelin Yüngülləşdirilməsi (Quantization) ⚖️

## 30.1. Quantization Nədir?

Bizim modelimiz təlim zamanı çəkilərini **32-bitlik (FP32)** və ya **16-bitlik (FP16)** dəqiqlikdə saxlayır. **Quantization (Kvantlaşdırma)** modelin çəkilərini daha az bit dəqiqliyinə (məsələn, **8-bit (Int8)** və ya **4-bit (Int4)**) çevirmə prosesidir.

**Niyə Quantization?**

1.  **Yaddaşa Qənaət:** Modelin ölçüsü kəskin şəkildə azalır. Məsələn, FP32-dən Int4-ə keçid modelin ölçüsünü **8 dəfə** azaldır.
2.  **Sürət:** Daha az bitlə işləmək GPU və ya CPU-da proqnozlaşdırma (inference) sürətini artırır.
3.  **RTX 2050 üçün Kritik:** Bizim modelimiz 134M parametrdir. FP32-də təxminən 536MB yer tutur. Int4-də isə cəmi **67MB** yer tutacaq. Bu, modelin Ollama-da yüngül və sürətli işləməsi üçün vacibdir.

## 30.2. Quantization Növləri

| Növ | Dəqiqlik | Faydası |
| :--- | :--- | :--- |
| **FP32** | 32-bit | Ən yüksək dəqiqlik. Təlim üçün standart. |
| **FP16** | 16-bit | Dəqiqlikdə az itki ilə yaddaşı 2 dəfə azaldır. |
| **Int8** | 8-bit | Yaddaşı 4 dəfə azaldır. Kiçik dəqiqlik itkisi ola bilər. |
| **Int4** | 4-bit | Yaddaşı 8 dəfə azaldır. **Bizim hədəfimiz.** |

## 30.3. GGUF Formatına Giriş

Biz modelimizi **GGUF (GPT-GEneration.cpp Unified Format)** formatına çevirəcəyik.

*   **GGUF** **llama.cpp** layihəsi tərəfindən yaradılmışdır və LLM-ləri CPU və ya məhdud VRAM-lı GPU-larda (məsələn, sizin RTX 2050) effektiv şəkildə işlətmək üçün nəzərdə tutulmuşdur.
*   **Ollama** (Gün 34-də öyrənəcəyik) məhz GGUF formatını dəstəkləyir.

**GGUF-un Üstünlükləri:**

1.  **Quantization Dəstəyi:** Int4, Int5, Int8 kimi müxtəlif kvantlaşdırma səviyyələrini dəstəkləyir.
2.  **Bütün Məlumat Bir Faylda:** Modelin arxitekturası, çəkiləri və tokenizatoru tək bir `.gguf` faylında saxlanılır.

## 30.4. Günün Tapşırığı: Hazırlıq

Quantization prosesi mürəkkəbdir və bir neçə addımdan ibarətdir:

1.  **PyTorch Modelini Hugging Face Formatına Çevirmək** (Gün 31-32).
2.  **Hugging Face Modelini Llama.cpp-yə Çevirmək** (Gün 33).
3.  **Llama.cpp ilə GGUF-a Kvantlaşdırmaq** (Gün 33).

Bu günün tapşırığı, bu proses üçün lazım olan əsas kitabxanaları quraşdırmaqdır:

```bash
# Hugging Face Transformers kitabxanası
pip install transformers

# Llama.cpp ilə işləmək üçün əsas kitabxana
pip install llama-cpp-python
```

**Gündəlik Tapşırıq:** Yuxarıdakı kitabxanaları quraşdırın. Quantization-ın modelin ölçüsünü necə dəyişdiyini və bunun RTX 2050 üçün nə qədər vacib olduğunu bir daha nəzərdən keçirin.
