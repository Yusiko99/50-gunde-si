# Gün 34: Ollama-ya Giriş (Modelin Dağıtımı) 🌐

## 34.1. Ollama Platformasının Əhəmiyyəti

**Ollama** – Böyük Dil Modellərini (LLM) yerli kompüterdə (CPU və ya GPU) asanlıqla işə salmaq üçün nəzərdə tutulmuş bir platformadır. Ollama, Llama.cpp-nin gücündən istifadə edərək, kvantlaşdırılmış modelləri (GGUF formatında) yüngül və sürətli şəkildə işlədir.

**Məntiq:** Ollama, LLM-lərin istifadəsini sadələşdirən bir interfeys təmin edir. Bu, modelin təlimindən sonra onu son istifadəçiyə çatdırmaq üçün ən effektiv yoldur.

## 34.2. Ollama-nın Quraşdırılması

Ollama-nın quraşdırılması əməliyyat sistemindən asılıdır (Windows, macOS, Linux). Rəsmi saytdan (https://ollama.com/) uyğun quraşdırma faylı endirilməlidir.

**Quraşdırma Məntiqi:** Ollama quraşdırıldıqdan sonra, arxa planda işləyən bir server (adətən 11434 portunda) işə salır və bu server vasitəsilə modellərə müraciət etmək mümkün olur.

## 34.3. Modelin Ollama-ya İdxalı (Modelfile)

Ollama-ya model idxal etmək üçün **Modelfile** adlı xüsusi bir konfiqurasiya faylı tələb olunur. Bu fayl Ollama-ya modelin harada olduğunu və hansı parametrlərlə işə salınacağını bildirir.

**`Modelfile`**

```dockerfile
# 1. Modelin əsasını təyin etmək
# Bu, bizim GGUF faylımızdır.
FROM ./az_llm_100m_q4_0.gguf

# 2. Modelin adını təyin etmək
PARAMETER model_name az-llm-100m

# 3. Modelin təsvirini təyin etmək
PARAMETER description "Azərbaycan dilində sıfırdan təlim edilmiş 100M parametrli, 4-bit kvantlaşdırılmış LLM."

# 4. Modelin temperaturunu təyin etmək (Yaradıcılıq dərəcəsi)
# 0.8 yaxşı bir başlanğıcdır.
PARAMETER temperature 0.8

# 5. Modelin kontekst uzunluğunu təyin etmək
# Bizim modelimizdə block_size 256 idi.
PARAMETER num_ctx 256

# 6. Modelin davranışını təyin edən sistem təlimatı
PARAMETER system "Sən Azərbaycan dilində danışan, dostyana və məlumatlı bir süni intellekt köməkçisisən. Cavabların qısa və məntiqli olsun."
```

## 34.4. Modelin Qurulması Əmri

**Addım 1:** `Modelfile` və `az_llm_100m_q4_0.gguf` faylları eyni qovluqda yerləşdirilir.

**Addım 2:** Terminalda `ollama create` əmri icra edilir.

```bash
ollama create az-llm-100m -f Modelfile
```

*   **`ollama create`:** Yeni bir model yaradır.
*   **`az-llm-100m`:** Modelin Ollama daxilindəki adı.
*   **`-f Modelfile`:** Konfiqurasiya faylının adını göstərir.

**Məntiq:** Bu əmr GGUF faylını Ollama-nın daxili yaddaşına köçürür və modelin parametrlərini tətbiq edir. Model artıq yerli kompüterdə istifadə üçün hazırdır.
