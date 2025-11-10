# Gün 34: Ollama-ya Giriş (Modelin Dağıtımı) 🌐

## 34.1. Ollama Nədir?

**Ollama** – Böyük Dil Modellərini (LLM) yerli kompüterinizdə (CPU və ya GPU) asanlıqla işə salmaq üçün nəzərdə tutulmuş bir platformadır. Ollama, Llama.cpp-nin gücündən istifadə edərək, kvantlaşdırılmış modelləri (GGUF formatında) çox yüngül və sürətli şəkildə işlədir.

**Ollama-nın Faydaları:**

1.  **Sadəlik:** Tək bir əmrlə modelinizi işə salmağa imkan verir.
2.  **GGUF Dəstəyi:** Bizim kvantlaşdırdığımız GGUF formatını dəstəkləyir.
3.  **API:** Modelinizi yerli bir API (Application Programming Interface) vasitəsilə istifadə etməyə imkan verir.

## 34.2. Ollama-nın Quraşdırılması

Sizin əməliyyat sisteminiz **Windows** olduğu üçün, Ollama-nın rəsmi saytından (https://ollama.com/) Windows üçün quraşdırma faylını endirməlisiniz.

**Quraşdırma Addımları:**

1.  Ollama-nın rəsmi saytına daxil olun.
2.  Windows üçün quraşdırma faylını endirin.
3.  Faylı icra edin və quraşdırmanı tamamlayın.

Quraşdırma tamamlandıqdan sonra, Ollama arxa planda işləyəcək və terminalda `ollama` əmri əlçatan olacaq.

## 34.3. Modelin Ollama-ya İdxalı (Import)

Bizim məqsədimiz **`az_llm_100m_q4_0.gguf`** faylını Ollama-ya idxal etməkdir. Bunun üçün **Modelfile** adlı xüsusi bir fayl yaratmalıyıq.

**Modelfile** – Ollama-ya modelin harada olduğunu, necə adlandırılacağını və hansı parametrlərlə işə salınacağını deyən konfiqurasiya faylıdır.

**`Modelfile`**

```dockerfile
# 1. Modelin əsasını təyin etmək
# Bu, bizim GGUF faylımızdır.
FROM ./az_llm_100m_q4_0.gguf

# 2. Modelin adını təyin etmək
# Bu adla modelə müraciət edəcəyik.
PARAMETER model_name az-llm-100m

# 3. Modelin təsvirini təyin etmək
# Ollama-da modelin təsviri
PARAMETER description "Azərbaycan dilində sıfırdan təlim edilmiş 100M parametrli LLM."

# 4. Modelin temperaturunu təyin etmək (Yaradıcılıq dərəcəsi)
# 0.8 yaxşı bir başlanğıcdır.
PARAMETER temperature 0.8

# 5. Modelin kontekst uzunluğunu təyin etmək
# Bizim modelimizdə block_size 256 idi.
PARAMETER num_ctx 256
```

## 34.4. Modelin Qurulması

**Addım 1: Modelfile-ı Yaratmaq**
Yuxarıdakı mətni **`Modelfile`** adlı bir fayla yadda saxlayın. Bu fayl və **`az_llm_100m_q4_0.gguf`** faylı eyni qovluqda olmalıdır.

**Addım 2: Ollama Build Əmrini İcra Etmək**

Terminalda bu qovluğa daxil olun və əmri icra edin:

```bash
ollama create az-llm-100m -f Modelfile
```

*   **`ollama create`:** Yeni bir model yaradır.
*   **`az-llm-100m`:** Modelin adı.
*   **`-f Modelfile`:** Konfiqurasiya faylının adını göstərir.

Bu əmr GGUF faylını Ollama-nın daxili yaddaşına köçürəcək və modelinizi istifadəyə hazır vəziyyətə gətirəcək.

**Gündəlik Tapşırıq:** Ollama-nı quraşdırın. `Modelfile` faylını yaradın və modelinizi Ollama-ya idxal etməyə hazırlaşın.
