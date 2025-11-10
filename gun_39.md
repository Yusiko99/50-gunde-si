# Gün 39: Modelin İdarə Edilməsi və Sürətləndirilməsi ⚙️

## 39.1. Modelin Versiyalaşdırılması (Versioning)

Model təkmilləşdirildikcə, onun müxtəlif versiyaları yaranır. Bu versiyaları idarə etmək üçün **Versiyalaşdırma (Versioning)** mexanizmi tətbiq olunur.

**Məntiq:** Versiyalaşdırma, hansı modelin hansı məlumat və parametrlərlə təlim edildiyini izləməyə, həmçinin ən yaxşı nəticə verən versiyaya geri qayıtmağa imkan verir.

**Versiyalaşdırma Üsulları:**

1.  **Fayl Adları:** `az_llm_v1.0_q4_0.gguf`, `az_llm_v1.1_q4_0.gguf`.
2.  **GitHub Tag-ləri:** Hər yeni versiya üçün GitHub-da **Tag** (etiket) yaratmaq.
3.  **Hugging Face Hub:** HF Hub avtomatik versiyalaşdırma və model kartları vasitəsilə versiyaların sənədləşdirilməsini təmin edir.

## 39.2. Proqnozlaşdırmanın Sürətləndirilməsi (Inference Optimization)

Modelin təlimi qədər, onun istifadəsi (proqnozlaşdırma) də sürətli olmalıdır.

### A. KV Cache (Key-Value Cache)

LLM-lər mətn yaradarkən hər dəfə yeni bir token proqnozlaşdırırlar. **KV Cache** mexanizmi, əvvəlki tokenlərin **Key** və **Value** matrislərini yaddaşda saxlayır.

*   **Məntiq:** Hər yeni token proqnozlaşdırılarkən, əvvəlki tokenlərin hesablanmış Key və Value matrislərini yenidən hesablamağa ehtiyac qalmır. Bu, proqnozlaşdırma sürətini kəskin şəkildə artırır. Ollama və Llama.cpp bu mexanizmi avtomatik tətbiq edir.

### B. Batching

Eyni anda bir neçə sorğuya cavab vermək üçün **Batching** istifadə olunur.

*   **Məntiq:** Bir neçə sorğunu birləşdirib eyni anda GPU-ya göndərmək, GPU-nun paralel hesablama gücündən daha effektiv istifadə etməyə imkan verir.

## 39.3. Ollama ilə Modelin İdarə Edilməsi

Ollama, yerli model idarəetməsi üçün sadə terminal əmrləri təklif edir:

| Əmr | Məqsəd |
| :--- | :--- |
| **`ollama list`** | Quraşdırılmış bütün modellərin siyahısını göstərir. |
| **`ollama rm model_name`** | Modeli sistemdən silir. |
| **`ollama pull model_name`** | Başqasının modelini Ollama Hub-dan endirir. |

**Nəticə:** Modelin idarə edilməsi və sürətləndirilməsi, təlim prosesinin tamamlayıcı hissəsidir və modelin real tətbiqlərdə effektivliyini təmin edir.
