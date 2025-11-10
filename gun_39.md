# Gün 39: Modelin İdarə Edilməsi və Sürətləndirilməsi ⚙️

## 39.1. Modelin İdarə Edilməsi (Versioning)

Modelinizi təkmilləşdirdikcə, onun müxtəlif versiyaları yaranacaq. Bu versiyaları idarə etmək üçün **Versioning (Versiyalaşdırma)** vacibdir.

**Versiyalaşdırma Üsulları:**

1.  **Fayl Adları:** Hər versiyanı fərqli adla saxlamaq. Məsələn: `az_llm_v1.0_q4_0.gguf`, `az_llm_v1.1_q4_0.gguf`.
2.  **GitHub Tag-ləri:** GitHub-da hər yeni versiya üçün bir **Tag** (etiket) yaratmaq.
3.  **Hugging Face Hub:** HF Hub hər model üçün avtomatik versiyalaşdırma təmin edir.

## 39.2. Proqnozlaşdırmanın Sürətləndirilməsi (Inference Optimization)

Təlimi bitirdik, lakin modelin istifadəsi (proqnozlaşdırma) də sürətli olmalıdır.

### A. Kvantlaşdırma (Quantization)

Gün 30-da öyrəndiyimiz kimi, **Int4 GGUF** formatı modelin ölçüsünü kəskin şəkildə azaldır və bu, proqnozlaşdırma sürətini artırır.

### B. Batching

Əgər siz eyni anda bir neçə sorğuya cavab vermək istəyirsinizsə, **Batching** istifadə edin.

*   **Nədir?** Bir neçə sorğunu (məsələn, 4 sorğu) birləşdirib eyni anda modelə göndərmək.
*   **Faydası:** GPU-nun paralel hesablama gücündən daha effektiv istifadə etməyə imkan verir.

### C. KV Cache (Key-Value Cache)

LLM-lər mətn yaradarkən hər dəfə yeni bir token proqnozlaşdırırlar. Hər yeni token üçün bütün əvvəlki tokenləri yenidən emal etmək əvəzinə, **KV Cache** əvvəlki tokenlərin **Key** və **Value** matrislərini yaddaşda saxlayır.

*   **Faydası:** Proqnozlaşdırma sürətini kəskin şəkildə artırır. Ollama və Llama.cpp bu mexanizmi avtomatik olaraq tətbiq edir.

## 39.3. Ollama ilə Modelin İdarə Edilməsi

Ollama model versiyalarını idarə etmək üçün sadə əmrlər təklif edir:

| Əmr | Məqsəd |
| :--- | :--- |
| **`ollama list`** | Kompüterinizdə quraşdırılmış bütün modellərin siyahısını göstərir. |
| **`ollama rm az-llm-100m`** | Modeli sistemdən silir. |
| **`ollama pull model_name`** | Başqasının modelini Ollama Hub-dan endirir. |

**Gündəlik Tapşırıq:** Ollama-nın `ollama list` əmrini icra edin. Modelinizin sürətini yoxlamaq üçün eyni sorğunu 5 dəfə göndərin və cavab vaxtlarını müqayisə edin.
