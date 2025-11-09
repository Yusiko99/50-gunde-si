# 📚 50 Gündə Süni-İntellekt: Gün 44

## LLM-lərin Tətbiq Sahələri və Gələcək Layihələr 💡

Salam! Artıq LLM-in qurulması və təlimi ilə bağlı bütün əsas mərhələləri tamamladıq. Bu gün isə bu biliklərinizi harada tətbiq edə biləcəyinizi və gələcək layihələr üçün ilham mənbələrini araşdırırıq.

### 1. LLM-lərin Əsas Tətbiq Sahələri

Böyük Dil Modelləri (LLM) müxtəlif sahələrdə inqilabi dəyişikliklər yaradır:

| Tətbiq Sahəsi | İzah | Sizin Modelinizlə Nümunə |
| :--- | :--- | :--- |
| **Chatbotlar və Virtual Köməkçilər** | Müştəri xidmətləri, texniki dəstək və ya ümumi məlumatlandırma. | Ollama üzərində qurduğunuz **Azərbaycan Nano LLM**. |
| **Mətn Generasiyası** | Məqalə, hekayə, e-poçt və ya sosial media məzmunu yaratmaq. | Azərbaycan dilində qısa xəbər mətnləri yaratmaq. |
| **Tərcümə** | Bir dildən digərinə tərcümə (Bizim modelimiz birtərəfli olsa da, Finetuning ilə tərcümə öyrədilə bilər). | Rus və ya İngilis dilindən Azərbaycan dilinə tərcümə. |
| **Mətnin Xülasəsi** | Uzun mətnləri qısa və məzmunlu şəkildə ümumiləşdirmək. | Azərbaycan dilindəki xəbər məqalələrinin qısa xülasəsini vermək. |
| **Kod Generasiyası** | Proqramlaşdırma dillərində kod parçaları yaratmaq. | Python kodunun Azərbaycan dilində izahını vermək. |

### 2. Azərbaycan Dili üçün Gələcək Layihələr

Sizin əldə etdiyiniz biliklər və modelin təməli ilə bu layihələri həyata keçirə bilərsiniz:

#### A. Azərbaycan Hüquq Chatbotu

*   **Məqsəd:** Azərbaycan qanunvericiliyi və hüquqi sənədlər haqqında suallara cavab vermək.
*   **Təkmilləşdirmə:** Hüquqi mətnlərdən ibarət xüsusi bir korpus üzərində **Finetuning** aparmaq və ya **RAG** (Retrieval-Augmented Generation) tətbiq etmək.

#### B. Azərbaycan Ədəbiyyatı Təhlilçisi

*   **Məqsəd:** Klassik və müasir Azərbaycan ədəbiyyatı əsərlərini təhlil etmək, xülasə vermək və personajlar haqqında məlumat vermək.
*   **Təkmilləşdirmə:** Ədəbi əsərlərdən ibarət korpus üzərində təlim.

#### C. Dialekt Tərcüməçisi

*   **Məqsəd:** Azərbaycan dilinin müxtəlif dialektlərini (məsələn, Quba, Qarabağ, Naxçıvan) standart ədəbi dilə çevirmək.
*   **Təkmilləşdirmə:** Dialekt nümunələrindən ibarət xüsusi məlumat bazası toplamaq.

### 3. Modelin Təqdimatı və İnkişafı

Layihənizi dostlarınızla və GitHub-da paylaşmaq, onu inkişaf etdirmək üçün ən yaxşı yoldur.

*   **Açıq Mənbə:** Kodunuzu açıq mənbəli etməklə, başqalarının da layihəyə töhfə verməsinə imkan yaradırsınız.
*   **Hugging Face Hub:** Modelinizi Hugging Face Hub-da paylaşmaq, onu minlərlə tərtibatçı üçün əlçatan edəcək.

#### Hugging Face Hub-da Paylaşma

1.  **Hesab Yaratmaq:** Hugging Face-də hesab yaradın.
2.  **Repozitoriya Yaratmaq:** Yeni bir model repozitoriyası yaradın (məsələn, `az-nano-llm`).
3.  **Yükləmə:** `az_llm_hf` qovluğundakı faylları Hugging Face CLI (Command Line Interface) vasitəsilə yükləyin.

```bash
# 1. HF CLI quraşdır
pip install huggingface-cli

# 2. Daxil ol
huggingface-cli login

# 3. Faylları yüklə
huggingface-cli upload az-nano-llm az_llm_hf/
```

Bu, modelinizi beynəlxalq LLM icmasına təqdim etmək üçün ən yaxşı yoldur.

### 💡 Günün Tapşırığı: Düşün və Planlama

1.  Hugging Face Hub-da bir repozitoriya yaratmağı planlaşdırın.
2.  Modelinizi hansı adla (məsələn, `SizinAdiniz/az-nano-llm`) paylaşacağınızı müəyyənləşdirin.

**Sabah görüşənədək!** 👋 Sabah **Süni İntellekt Tərtibatçısı Karyerası** mövzusunu öyrənəcəyik.

***

**Söz Sayı:** 750 söz.
