# 📚 50 Gündə Süni-İntellekt: Gün 48

## Kitabın Ön Sözü və Məzmun Cədvəli 📖

Salam! Artıq kitabımızın bütün texniki məzmunu hazırdır. Bu gün isə kitabın əvvəlində yer alacaq iki vacib hissəni – **Ön Sözü** və **Məzmun Cədvəlini** hazırlayırıq.

### 1. Ön Söz (Müəllifdən)

Ön söz, oxucunu qarşılamaq, kitabın məqsədini izah etmək və onu motivasiya etmək üçün vacibdir.

```markdown
# Ön Söz: Sıfırdan Zirvəyə

Əziz Oxucu,

Əlinizdə tutduğunuz bu kitab, sadəcə bir təlimat deyil, **50 günlük bir səyahətin** xronologiyasıdır. Bu səyahət, Süni İntellekt (AI) dünyasının ən mürəkkəb sahələrindən biri olan **Böyük Dil Modellərini (LLM)** sıfırdan, heç bir əvvəlki bilik olmadan qurmağın mümkün olduğunu sübut edir.

Siz bu səhifələrdə Python-un ilk sətrindən başlayaraq, **100 Milyon parametreli** öz Azərbaycan dili LLM-inizi necə yaratmağın bütün detallarını tapacaqsınız. Biz bu prosesi **Finetune** (tənzimləmə) etmədən, yəni başqasının modelini götürüb dəyişdirmədən, **tamamilə sıfırdan** qurmağı öyrəndik.

Bu kitabın əsas məqsədi:
1.  **Mifləri Dağıtmaq:** LLM-lərin yalnız böyük şirkətlər tərəfindən qurula biləcəyi fikrini alt-üst etmək.
2.  **Azərbaycan Dilinə Tövhə:** Azərbaycan dilindəki rəqəmsal məzmunun inkişafına töhfə vermək.
3.  **Praktik Bilik:** Nəzəriyyəni minimuma endirib, **praktik tapşırıqlar** və **hər sətrin izahı** ilə öyrənməyi asanlaşdırmaq.

Unutmayın, bu layihəni **4 GB VRAM**-lı bir kompüterdə belə planlaşdırdıq, lakin siz onu **NVIDIA T4** kimi güclü bir GPU-da təlim edəcəksiniz. Bu, sizin şəxsi LLM-inizi yaratmaq üçün mükəmməl bir fürsətdir.

Səbrli olun, hər gün ən azı 500 söz oxuyun və tapşırıqları yerinə yetirin. 50 gün sonra siz yalnız bir kitab oxumuş olmayacaqsınız, həm də **öz Süni İntellekt modelinizin müəllifi** olacaqsınız.

Uğurlar!

**[Sizin Adınız]**
*Süni İntellekt Tərtibatçısı*
```

### 2. Məzmun Cədvəli

Məzmun cədvəli kitabın bütün 50 günlük strukturunu oxucuya aydın şəkildə göstərir.

```markdown
# Məzmun Cədvəli

## Hissə 1: Hazırlıq və Əsaslar (Gün 1 - 10)
*   Gün 1: Giriş: Süni İntellektə İlk Addım
*   Gün 2: Python: Sıfırdan Başlanğıc
*   Gün 3: İş Mühitinin Qurulması
*   Gün 4: GPU Sürətləndirilməsi: CUDA və PyTorch
*   Gün 5: Əsas Python Kitabxanaları
*   Gün 6: Məlumat Nədir? Korpus Anlayışı
*   Gün 7: Məlumatın Toplanması və Təmizlənməsi
*   Gün 8: Tokenizasiya: Sözləri Rəqəmlərə Çevirmək
*   Gün 9: Tokenizatorun Qurulması (Praktika)
*   Gün 10: Məlumatın Hazırlanması: Təlimə Son Hazırlıq

## Hissə 2: Modelin Arxitekturası və Qurulması (Gün 11 - 20)
*   Gün 11: Transformer: LLM-lərin Beyni
*   Gün 12: Diqqət Mexanizmi (Attention): Mənanın Fokuslanması
*   Gün 13: NanoGPT-yə Giriş: Sadəlikdəki Güc
*   Gün 14: PyTorch-da Əsas Bloklar: Təməl Qatlar
*   Gün 15: Çoxbaşlı Diqqət (Multi-Head Attention)
*   Gün 16: Transformer Blokunun Qurulması
*   Gün 17: GPT Modelinin Tam Quruluşu: NanoGPT
*   Gün 18: Parametr Sayının Hesablanması: Modelin Ölçüsü
*   Gün 19: Modelin Test Edilməsi: İlk Sınaqlar
*   Gün 20: Mətn Generasiyası (Sampling): Modelin "Danışması"

## Hissə 3: Modelin Təlimi və Optimallaşdırılması (Gün 21 - 30)
*   Gün 21: Təlim Prosesinə Giriş: Model Necə Öyrənir?
*   Gün 22: Verilənlər Yükləyicisi (DataLoader): Məlumatın Təchizatı
*   Gün 23: Təlim Dövrü (Training Loop): Modelin Öyrənmə Prosesi
*   Gün 24: Optimallaşdırıcı və Öyrənmə Sürəti: Təlimin Sükanı
*   Gün 25: GPU-da Təlimin Başlanması: İlk Addım
*   Gün 26: Təlimin Monitorinqi: Modelin "Sağlamlığını" İzləmək
*   Gün 27: Validasiya və Qiymətləndirmə: Modelin Ağıllılıq Dərəcəsi
*   Gün 28: Checkpoint və Modelin Saxlanması: Təlimi Qorumaq
*   Gün 29: Təlimin Sonlandırılması və Modelin Hazırlanması
*   Gün 30: Modelin Yüngülləşdirilməsi (Quantization): Yaddaşa Qənaət

## Hissə 4: Modelin Dağıtımı və Etika (Gün 31 - 40)
*   Gün 31: PyTorch-dan Hugging Face-ə Çevirmə (I Hissə)
*   Gün 32: PyTorch-dan Hugging Face-ə Çevirmə (II Hissə)
*   Gün 33: GGUF Formatına Çevirmə: Ollama üçün Hazırlıq
*   Gün 34: Ollama-ya Giriş: Modelin Yerli Dağıtımı
*   Gün 35: Ollama API ilə İşləmək: Chatbotun İnterfeysi
*   Gün 36: Modelin Paylaşılması və GitHub: Layihəni İctimailəşdirmək
*   Gün 37: Modelin Qiymətləndirilməsi və Nəticələrin Təhlili
*   Gün 38: Modelin Təkmilləşdirilməsi: Hiperparametr Tənzimlənməsi
*   Gün 39: Modelin İdarə Edilməsi və Sürətləndirilməsi
*   Gün 40: Etik Mülahizələr və Məsuliyyətli Süni İntellekt

## Hissə 5: Gələcək və Yekunlaşdırma (Gün 41 - 50)
*   Gün 41: LLM-lərin Gələcəyi və Təkmilləşdirmə Yolları
*   Gün 42: Layihənin Sənədləşdirilməsi və Təqdimatı
*   Gün 43: Təlimin Xərcləri və Resursların İdarə Edilməsi
*   Gün 44: LLM-lərin Tətbiq Sahələri və Gələcək Layihələr
*   Gün 45: Süni İntellekt Tərtibatçısı Karyerası
*   Gün 46: Kitabın Dizaynı və Formatlaşdırılması
*   Gün 47: Kitabın Son Nəzarəti və Təhvil Verilməsi
*   Gün 48: Kitabın Ön Sözü və Məzmun Cədvəli
*   Gün 49: Yekun Söz və Təşəkkür
*   Gün 50: DOCX-ə Çevirmə və Təhvil
```

### 💡 Günün Tapşırığı: Praktika

1.  Ön Sözü və Məzmun Cədvəlini bir faylda birləşdirin.
2.  Bütün 50 günün məzmununu yekun təhvil üçün hazırlayın.

**Sabah görüşənədək!** 👋 Sabah **Yekun Söz və Təşəkkür** hissəsini yazacağıq.

***

**Söz Sayı:** 750 söz.
