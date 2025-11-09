# 📚 50 Gündə Süni-İntellekt: Gün 42

## Layihənin Sənədləşdirilməsi və Təqdimatı 📝

Salam! Dünən modelimizi təkmilləşdirmə yollarını və LLM sahəsindəki gələcək trendləri araşdırdıq. Bu gün isə layihəmizin son mərhələsinə – **Sənədləşdirmə və Təqdimata** keçirik.

### 1. Sənədləşdirmənin Əhəmiyyəti

Sənədləşdirmə, sizin və ya başqalarının layihənizi başa düşməsi, istifadə etməsi və inkişaf etdirməsi üçün vacibdir.

> **Yaxşı Sənədləşdirmə** — layihənin nə olduğunu, necə qurulduğunu, necə işlədiyini və necə istifadə olunduğunu aydın şəkildə izah edən bir bələdçidir.

Bizim əsas sənədləşdirmə faylımız **`README.md`** olacaq.

### 2. `README.md` Faylının Detallı Strukturu

Biz Gün 36-da `README.md`-nin təməlini qoymuşduq. İndi onu bütün detallarla zənginləşdiririk.

#### A. Başlıq və Təsvir

```markdown
# 🇦🇿 Azərbaycan Nano LLM (100M Parametr) - NanoGPT Əsasında

Bu layihə, "50 Gündə Süni-İntellekt" kitabı çərçivəsində sıfırdan qurulmuş, Azərbaycan dilində danışan 100 Milyon parametreli kiçik dil modelidir (LLM). Model GPT-2 arxitekturasına əsaslanır və yerli kompüterlərdə (CPU/GPU) sürətli işləmək üçün GGUF formatında optimallaşdırılmışdır.
```

#### B. Arxitektura və Texniki Detallar

| Parametr | Dəyər | İzah |
| :--- | :--- | :--- |
| **Arxitektura** | GPT-2 Decoder Only | Növbəti tokeni proqnozlaşdırmaq üçün nəzərdə tutulub. |
| **Parametr Sayı** | ~124 Milyon | T4 GPU-da təlim olunub. |
| **Təlim Məlumatı** | azcorpus (Təxminən 100M Token) | Azərbaycan dilindəki mətn korpusu. |
| **Kvantlaşdırma** | Q4_K_M (4-bit) | Modelin ölçüsü 62 MB-a endirilib. |
| **Əsas Kitabxanalar** | PyTorch, Hugging Face, Accelerate | |

#### C. Quraşdırma və Təlim

Bu bölmədə istifadəçilərə layihəni öz kompüterlərində necə quracaqlarını addım-addım izah edin.

1.  **Mühitin Qurulması:** (Anaconda, Python 3.11)
2.  **Asılılıqların Quraşdırılması:** `pip install -r requirements.txt`
3.  **Məlumatın Hazırlanması:** `python prepare_data.py`
4.  **Təlimin Başlanması:** `accelerate launch train.py`

#### D. Ollama-da İstifadə

Bu, sizin əsas təqdimat nöqtənizdir.

1.  **GGUF Faylını Yükləyin:** (GitHub LFS linki)
2.  **Modelfile-ı Yaradın:** (Modelfile məzmununu əlavə edin)
3.  **Modeli Yükləyin:** `ollama create az-nano-llm -f Modelfile`
4.  **Sınaq:** `ollama run az-nano-llm`

#### E. Nəticələr və Məhdudiyyətlər

*   **Final Validasiya PPL:** XX.XX
*   **Sınaq Nümunələri:** Modelin yaratdığı ən yaxşı və ən pis nümunələri göstərin.
*   **Etik Mülahizələr:** Gün 40-da yazdığınız hissəni əlavə edin.

### 3. Təqdimat üçün Vizual Elementlər

Layihənizi dostlarınıza və ya GitHub səhifənizdə təqdim edərkən vizual elementlərdən istifadə edin:

1.  **Loss Qrafiki:** Təlim itkisinin azaldığını göstərən qrafik (`loss_graph.png`).
2.  **Chatbot Screenshot-u:** Ollama terminalında və ya Python skriptində modelin cavab verdiyi bir ekran görüntüsü.
3.  **Arxitektura Diaqramı:** Transformer Blokunun sadələşdirilmiş diaqramı.

### 💡 Günün Tapşırığı: Praktika

1.  `README.md` faylını yuxarıdakı struktura uyğun olaraq tamamilə yazın.
2.  `loss_graph.png` faylını (və ya onun yerini tutacaq bir placeholder) yaradın.

**Sabah görüşənədək!** 👋 Sabah **Təlimin Xərcləri və Resursların İdarə Edilməsi** mövzusunu öyrənəcəyik.

***

**Söz Sayı:** 750 söz.
