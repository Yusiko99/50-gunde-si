# Gün 42: Layihənin Sənədləşdirilməsi və Təqdimatı 📝

## 42.1. Sənədləşdirmənin Əhəmiyyəti

Siz bu layihəni dostlarınızla və GitHub-da paylaşmaq istəyirsiniz. Yaxşı sənədləşdirmə (Documentation) layihənizin başqaları tərəfindən asanlıqla başa düşülməsi və istifadə edilməsi üçün kritikdir.

**Əsas Sənədləşdirmə Faylları:**

1.  **`README.md` (Əsas Təqdimat):** Layihənin qısa icmalı.
2.  **`INSTALL.md` (Quraşdırma Təlimatı):** Python, PyTorch, Ollama quraşdırma addımları.
3.  **`TRAINING.md` (Təlim Qeydləri):** Təlim zamanı istifadə olunan hiperparametrlər, Loss qrafikləri və RTX 2050 üçün optimallaşdırma qeydləri.

## 42.2. `TRAINING.md` Faylının Quruluşu

Bu fayl, layihənizin elmi hissəsini təşkil edir.

| Bölmə | Məzmun |
| :--- | :--- |
| **1. Model Arxitekturası** | 134M parametrli GPT-2 Decoder-only modelinin hiperparametrləri (n_embd=768, n_layer=12, n_head=12). |
| **2. Korpus** | Korpusun həcmi (məsələn, 1.2 GB xalis mətn), mənbələri (Vikipediya, Xəbərlər) və təmizləmə prosesi. |
| **3. Təlim Konfiqurasiyası** | **GPU:** NVIDIA RTX 2050 (4GB VRAM). **Optimallaşdırma:** FP16 Mixed Precision, Gradient Accumulation (4 addım). **Batch Size:** 4 (Effektiv Batch Size: 16). **Öyrənmə Sürəti:** 3e-4. |
| **4. Nəticələr** | Təlim və Validasiya Loss-unun qrafikləri. Ən yaxşı Validasiya Loss-u və PPL dəyəri. |
| **5. Nümunə Generasiya** | Modelin yaratdığı ən yaxşı və ən pis nümunələr. |

## 42.3. Təqdimat (Dostlarınız üçün)

Dostlarınıza layihənizi təqdim edərkən aşağıdakı 3 əsas məqama fokuslanın:

1.  **Problem:** Azərbaycan dilində güclü, açıq mənbəli LLM-lərin olmaması.
2.  **Həll Yolu:** Sıfırdan öz korpusumuzu toplayaraq 134M parametrli LLM yaratdıq.
3.  **Nəticə:** Modelimiz Ollama-da işləyir və yerli kompüterdə sürətli cavab verir.

**Gündəlik Tapşırıq:** `TRAINING.md` faylını yaradın və təlim zamanı topladığınız bütün məlumatları (təlim parametrləri, Loss dəyərləri) bu fayla daxil edin.
