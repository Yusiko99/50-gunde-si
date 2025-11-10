# 📚 50 Gündə Süni-İntellekt: Gün 6

## Məlumat Nədir? Korpus Anlayışı 🧠

Salam! Əsas alətlərimizi (Python, PyTorch, NumPy) quraşdırdıq. İndi isə LLM-in **qidasına** – **məlumatlara** keçirik. Unutmayın, modelimiz nə qədər keyfiyyətli məlumatla qidalanarsa, bir o qədər ağıllı olar.

### 1. Təbii Dilin Emalı (NLP)

Bizim layihəmiz **Təbii Dilin Emalı (NLP)** sahəsinə aiddir.

> **Təbii Dilin Emalı (NLP)** — kompüterlərin insan dilini (danışıq və ya yazı) başa düşməsi, təhlil etməsi və yaratması ilə məşğul olan Süni İntellekt sahəsidir.

LLM-lər NLP-nin ən mürəkkəb və ən güclü tətbiqlərindən biridir.

### 2. Mətn Korpusu (Text Corpus) Nədir?

LLM-i təlim etmək üçün ona çoxlu mətn verməliyik. Bu mətnlərin toplusu **Mətn Korpusu** adlanır.

> **Mətn Korpusu** — dil tədqiqatları və ya maşın öyrənməsi üçün toplanmış, strukturlaşdırılmış və təmizlənmiş böyük həcmli mətnlər toplusudur.

Korpus, modelimizin **dünyanı** necə görməsini və **Azərbaycan dilini** necə başa düşməsini müəyyən edir.

| Korpusun Xüsusiyyətləri | Niyə Vacibdir? |
| :--- | :--- |
| **Həcm** | Nə qədər böyük olsa, model o qədər çox söz və cümlə quruluşu öyrənər. |
| **Keyfiyyət** | Təmiz, səhvsiz mətnlər modelin səhv öyrənməsinin qarşısını alır. |
| **Müxtəliflik** | Xəbərlər, elmi məqalələr, bədii ədəbiyyat, dialoqlar kimi müxtəlif janrlar modelin bilik dairəsini genişləndirir. |

### 3. Azərbaycan Dili üçün Məlumat Bazası

Bizim ən böyük çətinliyimiz, Azərbaycan dilinin **"az resurslu dil"** olmasıdır. İngilis dili üçün terabaytlarla məlumat varkən, Azərbaycan dili üçün açıq mənbəli, təmizlənmiş məlumat tapmaq çətindir.

Ancaq, araşdırmamız nəticəsində tapdığımız **əsas mənbə** bizim üçün ideal başlanğıc nöqtəsidir:

#### 🌟 azcorpus: Azərbaycanın Ən Böyük Açıq Mənbəli Korpusu

**İdarəetmə Sistemləri İnstitutu (İSİ)** tərəfindən yaradılmış **azcorpus** bizim LLM layihəmizin təməlini təşkil edəcək.

| Xüsusiyyət | Dəyər | Əhəmiyyəti |
| :--- | :--- | :--- |
| **Həcm** | **1.9 Milyon** sənəd | Modelin ilkin təlimi üçün kifayət qədər böyük həcm. |
| **Cümlə Sayı** | **~18 Milyon** cümlə | Modelin qrammatik quruluşları öyrənməsi üçün əsas. |
| **Həcmi** | **23.4 GB** | Yüklənməsi və emalı üçün idarəolunan bir ölçü. |
| **Mənbələr** | Xəbər saytları, jurnallar, Vikipediya, kitablar. | Müxtəlif mövzuları (siyasət, iqtisadiyyat, elm) əhatə edir. |
| **Əlçatanlıq** | **Açıq Mənbəli (Open-Source)** | Pulsuz və sərbəst istifadə edilə bilər. |

**azcorpus**-u Hugging Face platformasında tapa bilərik: `https://huggingface.co/datasets/azcorpus/azcorpus_v0`
azcorpus-a alternativ olaraq daha kiçik ölçülü dataset axtarırsınızsa : `https://huggingface.co/datasets/Yusiko/AZE_friendly_dataset`

### 4. Hugging Face Datasets: Məlumatların Evidir

**Hugging Face** platforması Dərin Öyrənmə dünyasında inqilab edib. O, modelləri, tokenizatorları və ən əsası, **məlumat bazalarını** (Datasets) asanlıqla paylaşmağa imkan verir.

Biz **azcorpus**-u birbaşa Hugging Face kitabxanası vasitəsilə Python kodumuzda yükləyəcəyik.

#### Quraşdırma

`llm_50gun` mühitində Hugging Face `datasets` kitabxanasını quraşdıraq:

```bash
pip install datasets
```

#### Məlumatın Yüklənməsi Nümunəsi

Bu, sadəcə bir nümunədir. Sabah daha ətraflı izah edəcəyik.

```python
from datasets import load_dataset

# azcorpus-u Hugging Face-dən yükləyirik
dataset = load_dataset("azcorpus/azcorpus_v0")

# Yüklənmiş məlumatın strukturuna baxırıq
print(dataset)
```

**Kodun İzahı:**
*   `from datasets import load_dataset`: Məlumat bazalarını yükləmək üçün funksiyanı daxil edirik.
*   `load_dataset("azcorpus/azcorpus_v0")`: Hugging Face-dəki `azcorpus` məlumat bazasını yükləyir.

### 💡 Günün Tapşırığı: Praktika

1.  `llm_50gun` mühitində `datasets` kitabxanasını quraşdırın.
2.  Brauzerinizdə `https://huggingface.co/datasets/azcorpus/azcorpus_v0` linkinə daxil olun və məlumat bazasının tərkibini araşdırın.
3.  Özünüz üçün qeyd edin: LLM-in təlimi üçün **azcorpus**-dan başqa hansı mənbələrdən (məsələn, Azərbaycan Vikipediyası, rəsmi sənədlər) məlumat toplamaq olar?

**Sabah görüşənədək!** 👋 Sabah **Məlumatın Toplanması və Təmizlənməsi** prosesinə başlayacağıq. Məlumatı necə yükləyib, necə təmizləyəcəyimizi öyrənəcəyik.

***

**Söz Sayı:** 650 söz.
