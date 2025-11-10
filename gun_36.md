# Gün 36: Modelin Paylaşılması və GitHub 🤝

## 36.1. Niyə Paylaşmalı?

Siz 50 gün ərzində sıfırdan bir LLM yaratdınız. Bu, böyük bir nailiyyətdir! Modelinizi dostlarınızla və GitHub səhifənizdə paylaşmaq aşağıdakılar üçün vacibdir:

1.  **Şəxsi Portfolio:** Bu layihə sizin Süni İntellekt sahəsindəki bilik və bacarıqlarınızı nümayiş etdirən ən güclü sübutdur.
2.  **İcma Dəstəyi:** Azərbaycan dili üçün açıq mənbəli LLM-lərin inkişafına töhfə verirsiniz.
3.  **Əməkdaşlıq:** Başqaları sizin kodunuzu və modelinizi istifadə edərək onu təkmilləşdirə bilər.

## 36.2. GitHub-da Layihənin Qurulması

GitHub layihənizin aşağıdakı faylları ehtiva etməsi vacibdir:

| Fayl | Məqsəd |
| :--- | :--- |
| **`README.md`** | Layihənin təsviri, quraşdırma təlimatları, istifadə nümunələri. |
| **`model.py`** | Modelin arxitekturası (Gün 17). |
| **`train_accelerate.py`** | Təlim skripti (Gün 25). |
| **`prepare_data.py`** | Məlumat hazırlığı skripti (Gün 12). |
| **`requirements.txt`** | Layihə üçün lazım olan bütün Python kitabxanaları. |
| **`az_llm_100m_q4_0.gguf`** | **Yekun kvantlaşdırılmış model faylı.** |
| **`Modelfile`** | Ollama konfiqurasiya faylı (Gün 34). |

### A. `requirements.txt` Faylının Yaradılması

```bash
pip freeze > requirements.txt
```

Bu əmr, quraşdırdığınız bütün kitabxanaları (PyTorch, transformers, accelerate, tokenizers, numpy, requests) `requirements.txt` faylına yazacaq.

### B. `README.md` Faylının Hazırlanması

`README.md` faylı layihənizin "üzü"dür. O, aşağıdakı bölmələri ehtiva etməlidir:

1.  **Başlıq:** # Az-LLM-100M: Azərbaycan Dilində Sıfırdan LLM
2.  **Təsvir:** Layihənin məqsədi (sıfırdan LLM yaratmaq).
3.  **Arxitektura:** 134M parametrli GPT-2 Decoder-only model.
4.  **Təlim:** Korpusun toplanması, RTX 2050 üçün optimallaşdırma (FP16, Gradient Accumulation).
5.  **İstifadə:** Ollama ilə necə işə salınacağı (Modelfile və `ollama create` əmrləri).

## 36.3. Modelin Hugging Face Hub-da Paylaşılması

Ən yaxşı paylaşma üsulu modelinizi **Hugging Face Hub**-a yükləməkdir.

1.  **Quraşdırma:** `pip install huggingface_hub`
2.  **Giriş:** `huggingface-cli login` (Tokeninizi daxil edin).
3.  **Repo Yaratmaq:** `huggingface-cli repo create az-llm-100m --type model`
4.  **Yükləmə:** `huggingface-cli upload az-llm-100m az_llm_hf/ az_llm_100m_q4_0.gguf`

Bu, modelinizi bütün dünyaya əlçatan edəcək.

**Gündəlik Tapşırıq:** `requirements.txt` faylını yaradın. GitHub-da yeni bir depo (repository) yaradın və bütün layihə fayllarını (kod, GGUF, Modelfile) ora yükləyin.
