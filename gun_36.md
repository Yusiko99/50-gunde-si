# Gün 36: Modelin Paylaşılması və GitHub 🤝

## 36.1. Açıq Mənbəli Paylaşımın Əhəmiyyəti

LLM layihəsinin açıq mənbəli şəkildə paylaşılması, təkcə şəxsi portfolionu gücləndirmir, həm də dil icmasına və elmi tərəqqiyə töhfə verir.

**Məntiq:** Açıq mənbə, modelin şəffaflığını təmin edir, başqalarına modelin performansını yoxlamağa və onu təkmilləşdirməyə imkan yaradır.

## 36.2. GitHub Layihəsinin Quruluşu

GitHub deposu layihənin bütün komponentlərini ehtiva etməlidir.

| Fayl | Məqsəd |
| :--- | :--- |
| **`README.md`** | Layihənin təsviri, arxitekturası, təlim parametrləri və istifadə təlimatları. |
| **`requirements.txt`** | Layihənin işləməsi üçün lazım olan bütün Python kitabxanaları. |
| **`model.py`** | Modelin PyTorch arxitekturası (Gün 17). |
| **`train_accelerate.py`** | Təlim skripti (Gün 25). |
| **`prepare_data.py`** | Məlumat hazırlığı skripti (Gün 12). |
| **`az_llm_100m_q4_0.gguf`** | **Yekun kvantlaşdırılmış model faylı.** |
| **`Modelfile`** | Ollama konfiqurasiya faylı (Gün 34). |

### A. `requirements.txt` Faylının Yaradılması

Bu fayl, layihənin asılılıqlarını qeyd edir.

```bash
pip freeze > requirements.txt
```

### B. `README.md` Faylının Məzmunu

`README.md` faylı layihənin texniki sənədləşdirilməsinin əsasını təşkil edir.

1.  **Başlıq:** Modelin adı və qısa təsviri.
2.  **Arxitektura:** 134M parametrli GPT-2 Decoder-only model.
3.  **Təlim Korpusu:** Korpusun həcmi və mənbələri (sıfırdan toplanmış).
4.  **Optimallaşdırma:** Məhdud VRAM (4GB) üçün istifadə olunan texnikalar (FP16, Gradient Accumulation).
5.  **Dağıtım:** Ollama ilə istifadə təlimatları (`Modelfile` və `ollama create` əmrləri).

## 36.3. Hugging Face Hub-da Paylaşım

Modelin ən geniş auditoriyaya çatması üçün **Hugging Face Hub** istifadə olunur.

**Məntiq:** HF Hub, LLM-lər üçün mərkəzləşdirilmiş bir depo rolunu oynayır.

**Paylaşım Addımları:**

1.  **Quraşdırma:** `pip install huggingface_hub`
2.  **Giriş:** `huggingface-cli login` (Token daxil edilir).
3.  **Repo Yaratmaq:** `huggingface-cli repo create az-llm-100m --type model`
4.  **Yükləmə:** Bütün HF faylları (`config.json`, `tokenizer.json`, `pytorch_model.bin`) və GGUF faylı depoya yüklənir.

**Nəticə:** Modelin bütün komponentləri açıq şəkildə sənədləşdirilir və istifadəçilər tərəfindən asanlıqla əldə edilə bilər.
