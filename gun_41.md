# 📚 50 Gündə Süni-İntellekt: Gün 41

## LLM-lərin Gələcəyi və Təkmilləşdirmə Yolları 🔮

Salam! Artıq 40 günü geridə qoyduq. Siz sıfırdan bir LLM-in necə qurulduğunu, təlim olunduğunu və Ollama-da necə işə salındığını öyrəndiniz. Bu gün isə modelinizi gələcəkdə necə təkmilləşdirə biləcəyinizi və dil modelləri sahəsindəki ən son trendləri araşdırırıq.

### 1. Modelin Təkmilləşdirilməsi Yolları

Sizin modeliniz **Pre-trained** (Öncədən Təlim Olunmuş) bir modeldir. Onu daha da yaxşılaşdırmaq üçün bu yolları izləyə bilərsiniz:

#### A. Məlumatın Artırılması (Data Augmentation)

*   **Daha Çox Məlumat:** Ən əsas yol daha çox keyfiyyətli Azərbaycan dili mətni toplamaqdır. Məsələn, 100M token yerinə 1 Milyard token üzərində təlim etmək.
*   **Sintetik Məlumat:** Mövcud məlumatı tərcümə və ya paraphrasing (yenidən ifadə etmə) vasitəsilə süni şəkildə artırmaq.

#### B. Finetuning (Tənzimləmə)

Modeli ümumi mətn üzərində təlim etdikdən sonra, onu xüsusi bir tapşırıq üçün (məsələn, sual-cavab, dialoq) yenidən təlim etmək.

*   **Supervised Finetuning (SFT):** Modelə xüsusi formatda (Prompt: Cavab) nümunələr verilir.
*   **RLHF (Reinforcement Learning from Human Feedback):** Modelin cavabları insanlar tərəfindən qiymətləndirilir və model bu rəylərə əsasən öyrənir.

#### C. Modelin Ölçüsünün Artırılması

*   **Daha Böyük Model:** İmkan olduqda, modelin parametr sayını 100M-dən 300M və ya 7B-yə qədər artırmaq. Bu, modelin daha mürəkkəb əlaqələri öyrənməsinə imkan verəcək.

### 2. Ən Son Trendlər

LLM sahəsi çox sürətlə inkişaf edir. Gələcəkdə modelinizi bu trendlərə uyğunlaşdıra bilərsiniz:

#### A. RAG (Retrieval-Augmented Generation)

*   **Prinsip:** Model cavab verməzdən əvvəl, xüsusi bir məlumat bazasında (məsələn, Azərbaycan tarixi sənədləri) axtarış edir və cavabını bu məlumatlara əsasən formalaşdırır.
*   **Üstünlüyü:** Modelin bilik bazasını təlim etmədən yeniləməyə və **halüsinasiyaların** (yalan məlumat verməyin) qarşısını almağa kömək edir.

#### B. MoE (Mixture of Experts)

*   **Prinsip:** Modelin bəzi qatları bir neçə kiçik neyron şəbəkəsinə (Ekspertlərə) bölünür. Hər bir sorğu üçün yalnız ən uyğun Ekspertlər aktivləşdirilir.
*   **Üstünlüyü:** Modelin parametr sayı çox böyük olsa da (məsələn, 1 Trilyon), hər sorğu üçün yalnız kiçik bir hissəsi istifadə olunduğundan, sürətli və effektivdir.

#### C. Multi-Modallıq

*   **Prinsip:** Modelin təkcə mətnlə deyil, həm də şəkillər, səslər və videolarla işləməsi.
*   **Gələcək:** Sizin modelinizə Azərbaycan dilində şəkilləri təsvir etməyi və ya səsləri anlamağı öyrətmək.

### 3. Təkmilləşdirmə üçün Praktik Addım: Yeni Tokenizator

Bizim BPE tokenizatorumuz yaxşı bir başlanğıcdır. Lakin, Hugging Face-in **SentencePiece** tokenizatoru daha müasir və effektivdir.

**Gələcək Tapşırıq:**
1.  `SentencePiece` quraşdırın.
2.  `azcorpus` üzərində yeni bir `SentencePiece` tokenizatoru təlim edin.
3.  Modelinizi bu yeni tokenizatorla yenidən təlim edin.

### 💡 Günün Tapşırığı: Düşün və Planlama

1.  Modelinizi hansı sahədə (məsələn, hüquq, tibb, ədəbiyyat) ixtisaslaşdırmaq istərdiniz?
2.  Bu ixtisaslaşma üçün hansı növ məlumatlara ehtiyacınız olacaq?

**Sabah görüşənədək!** 👋 Sabah **Layihənin Sənədləşdirilməsi və Təqdimatı** mövzusunu öyrənəcəyik.

***

**Söz Sayı:** 750 söz.
