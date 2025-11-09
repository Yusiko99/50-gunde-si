# 📚 50 Gündə Süni-İntellekt: Gün 1

## Giriş: Süni İntellektə İlk Addım 🚀

Salam, əziz oxucu!

Bu gün, **Süni İntellekt (Sİ)** dünyasına atdığımız böyük səyahətin ilk günüdür. 50 gün ərzində biz birlikdə sıfırdan başlayaraq, **Azərbaycan dilində danışa bilən** və sizinlə ünsiyyət qura bilən bir chatbotun – yəni **Böyük Dil Modelinin (LLM)** – necə yaradıldığını öyrənəcəyik. Bu, çətin, lakin inanılmaz dərəcədə maraqlı bir macəra olacaq!

Unutmayın, bu kitab sizin Python biliyinizin olmadığını nəzərə alaraq yazılıb. Hər şeyi ən sadə dildə, addım-addım izah edəcəyəm.

### 1. Süni İntellekt (Sİ) Nədir?

Təsəvvür edin ki, bir kompüter proqramı insan kimi düşünə, öyrənə və qərar verə bilir. Bax, bu, sadə dildə **Süni İntellekt** deməkdir.

> **Süni İntellekt (Sİ)** — maşınların insan zəkasına xas olan funksiyaları (öyrənmə, problem həlli, qərar qəbul etmə) yerinə yetirməsini təmin edən elm sahəsidir.

Sİ-nin üç əsas qolu var ki, bizim layihəmiz onlardan ikisi ilə sıx bağlıdır:

| Qol | Məqsəd | Misal |
| :--- | :--- | :--- |
| **Maşın Öyrənməsi (Machine Learning - ML)** | Maşınların açıq şəkildə proqramlaşdırılmadan, verilənlər (data) əsasında öyrənməsini təmin etmək. | Spam filtrləri, məhsul tövsiyələri. |
| **Dərin Öyrənmə (Deep Learning - DL)** | İnsan beyninin neyron şəbəkələrinə bənzəyən **dərin neyron şəbəkələri** istifadə edərək mürəkkəb öyrənməni həyata keçirmək. | Şəkil tanıma, bizim LLM-imiz. |
| **Təbii Dilin Emalı (Natural Language Processing - NLP)** | Kompüterlərin insan dilini (Azərbaycan dili, İngilis dili və s.) başa düşməsi, təhlil etməsi və yaratması. | Tərcümə proqramları, chatbotlar. |

Bizim 50 günlük layihəmizdə biz **Dərin Öyrənmə** metodlarından istifadə edərək **Təbii Dilin Emalı** sahəsinə aid olan bir model – **LLM** – yaradacağıq.

### 2. Böyük Dil Modeli (LLM) Nədir?

**LLM** (Large Language Model) sözlərin böyük bir hissəsini təşkil edir. Bu, sadəcə bir chatbot deyil, mətnlə işləmək üçün nəhəng bir beyindir.

Təsəvvür edin ki, sizə minlərlə kitab oxuyan bir dostunuz var. O, hansı sözdən sonra hansı sözün gəlmə ehtimalının daha yüksək olduğunu dəqiq bilir. LLM də eynilə belə işləyir. O, milyonlarla, hətta milyardlarla sözdən ibarət mətnləri oxuyur və öyrənir ki, bir cümləni necə davam etdirmək lazımdır.

> **LLM-in Əsas İş Prinsipi:** Növbəti sözü (tokeni) proqnozlaşdırmaq.

Məsələn, modelə "Mən bu gün Bakıda" yazsanız, o, öyrəndiyi məlumatlara əsaslanaraq ən ehtimal olunan növbəti sözləri təklif edəcək: "gəzirəm", "oldum", "işləyirəm" və s.

### 3. Niyə Sıfırdan Başlayırıq?

Siz soruşa bilərsiniz: "Hazır modellər varkən, niyə sıfırdan LLM yaradaq?" Bunun bir neçə **əsas səbəbi** var:

1.  **Öyrənmə:** Sıfırdan başlamaq, LLM-in hər bir hissəsinin necə işlədiyini, hər bir kod sətrinin nə demək olduğunu dərindən başa düşməyə kömək edir. Bu, sadəcə istifadəçi olmaqdan, **yaradıcı** olmağa keçiddir.
2.  **Azərbaycan Dili:** Azərbaycan dili **"az resurslu dil"** hesab olunur. Bu o deməkdir ki, İngilis dili kimi dillər üçün nəhəng məlumat bazaları varkən, Azərbaycan dili üçün keyfiyyətli və böyük məlumat bazası tapmaq çətindir. Sıfırdan başlamaq, modelimizi məhz bizim dilimizin incəliklərinə uyğunlaşdırmağa imkan verir.
3.  **Yüngüllük və Sürət:** Bizim hədəfimiz **100 Milyon parametreli** yüngül, lakin güclü bir model yaratmaqdır. Bu cür kiçik modellər sizin **NVIDIA T4 (12 GB VRAM)** kimi şəxsi cihazınızda və ya Ollama kimi yüngül platformalarda sürətlə işləyə bilər.

### 4. Layihəmizin Texniki Özəllikləri

| Xüsusiyyət | Dəyər | Niyə bu seçildi? |
| :--- | :--- | :--- |
| **Parametr Sayı** | **~100 Milyon** | Yüngül, sürətli və şəxsi kompüterdə təlim üçün uyğundur. |
| **Arxitektura** | **NanoGPT (GPT-2 əsaslı)** | Sadə, başa düşülən və sıfırdan tətbiq etmək üçün ideal. |
| **Dil** | **Azərbaycan Dili** | Modelin məhz bizim dilimizdə yüksək performans göstərməsini təmin etmək. |
| **Əsas Çərçivə** | **PyTorch** | Dərin Öyrənmə üçün ən populyar və çevik kitabxana. |
| **Dağıtım Formatı** | **GGUF** | Ollama və digər yüngül platformalarda istifadə üçün optimallaşdırılmış format. |

### 💡 Günün Tapşırığı: Düşün və Araşdır

Bu gün heç bir kod yazmırıq. Sadəcə düşünürük:

1.  **Süni İntellekt** haqqında eşitdiyiniz ən maraqlı şeyi xatırlayın.
2.  **Azərbaycan dilində** bir chatbotun hansı işləri görməsini istərdiniz? (Məsələn, lüğət tapşırıqları, tarix haqqında məlumat vermək).

Bu sualların cavabları bizim gələcək modelimizin **"şəxsiyyətini"** formalaşdırmağa kömək edəcək.

**Sabah görüşənədək!** 👋 Sabah **Python**-u necə quraşdıracağımızı öyrənəcəyik. Bu, bizim ilk **praktiki** addımımız olacaq.

***

**Söz Sayı:** 600 söz.
