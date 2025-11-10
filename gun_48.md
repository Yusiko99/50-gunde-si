# Gün 48: Kitabın Ön Sözü və Məzmun Cədvəli 📖

## 48.1. Ön Söz

**50 Gündə Süni-İntellekt: Azərbaycan Dilində LLM-i Sıfırdan Qurmaq**

Bu kitab, Süni İntellekt (Sİ) dünyasına atılan cəsarətli bir addımdır. Əgər siz bu sətirləri oxuyursunuzsa, deməli, sadəcə bir chatbot istifadəçisi olmaqla kifayətlənmir, onun necə işlədiyini dərindən anlamaq istəyirsiniz.

Bu, sadəcə bir təlimat kitabı deyil, **50 günlük praktik səyahətin** yol xəritəsidir. Biz bu səyahətə Python-un sıfırdan quraşdırılmasından başlayıb, Azərbaycan dilində öz korpusumuzu toplamaq, 134 Milyon parametrli **Transformer** modelini PyTorch-da sıfırdan kodlamaq və nəhayət, kvantlaşdırılmış **GGUF** modelimizi **Ollama** platformasında işə salmaqla yekunlaşdıracağıq.

Bu kitabın ən böyük özəlliyi, məhdud resurslarla (4GB VRAM-lı RTX 2050) belə, böyük işlər görməyin mümkünlüyünü göstərməsidir. Hər bir kod sətri, hər bir nəzəri anlayış sadə və anlaşılan Azərbaycan dilində izah edilmişdir.

Bu kitabı bitirdikdən sonra, siz sadəcə bir LLM yaratmış olmayacaqsınız; siz Süni İntellektin əsas prinsiplərini mənimsəmiş, bu sahədəki biliklərinizi sübut edəcək bir layihəyə sahib olmuş olacaqsınız.

Uğurlar!

## 48.2. Məzmun Cədvəli

| Hissə | Mövzu | Günlər |
| :--- | :--- | :--- |
| **Hissə 1** | **Hazırlıq və Məlumat Mühəndisliyi** | Gün 1 - Gün 12 |
| | Gün 1: Giriş: Süni İntellektə İlk Addım | |
| | Gün 2: Python: Sıfırdan Başlanğıc | |
| | Gün 3: İş Mühitinin Qurulması | |
| | Gün 4: GPU Sürətləndirilməsi: RTX 2050 üçün Optimallaşdırma | |
| | Gün 5: Əsas Python Kitabxanaları | |
| | Gün 6: Dataset İnşası I: Məlumat Mənbələrinin Təyini | |
| | Gün 7: Dataset İnşası II: Web Scraping (Məlumatın Çəkilməsi) | |
| | Gün 8: Dataset İnşası III: Məlumatın Təmizlənməsi (Cleaning) | |
| | Gün 9: Dataset İnşası IV: Məlumatın Normallaşdırılması | |
| | Gün 10: Tokenizasiya I: Sözləri Rəqəmlərə Çevirmək | |
| | Gün 11: Tokenizasiya II: Tokenizatorun Qurulması (Praktika) | |
| | Gün 12: Məlumatın Hazırlanması: Rəqəmləşdirmə | |
| **Hissə 2** | **Modelin Arxitekturası və Qurulması** | Gün 13 - Gün 20 |
| | Gün 13: Transformer: LLM-lərin Beyni | |
| | Gün 14: Diqqət Mexanizmi (Attention) | |
| | Gün 15: Çoxbaşlı Diqqət (Multi-Head Attention) | |
| | Gün 16: Transformer Blokunun Qurulması | |
| | Gün 17: GPT Modelinin Tam Quruluşu | |
| | Gün 18: Parametr Sayının Hesablanması | |
| | Gün 19: Modelin Test Edilməsi (Generation) | |
| | Gün 20: Mətn Generasiyası (Sampling) | |
| **Hissə 3** | **Təlim və Optimallaşdırma** | Gün 21 - Gün 30 |
| | Gün 21: Təlim Prosesinə Giriş | |
| | Gün 22: Verilənlər Yükləyicisi (DataLoader) | |
| | Gün 23: Təlim Dövrü (Training Loop) | |
| | Gün 24: Optimallaşdırıcı və Öyrənmə Sürəti | |
| | Gün 25: RTX 2050-də Təlimin Başlanması (Optimallaşdırma) | |
| | Gün 26: Təlimin Monitorinqi | |
| | Gün 27: Validasiya və Qiymətləndirmə | |
| | Gün 28: Checkpoint və Modelin Saxlanması | |
| | Gün 29: Təlimin Sonlandırılması və Modelin Hazırlanması | |
| | Gün 30: Modelin Yüngülləşdirilməsi (Quantization) | |
| **Hissə 4** | **Dağıtım və Paylaşım** | Gün 31 - Gün 40 |
| | Gün 31: PyTorch-dan Hugging Face-ə Çevirmə (I Hissə) | |
| | Gün 32: PyTorch-dan Hugging Face-ə Çevirmə (II Hissə) | |
| | Gün 33: GGUF Formatına Çevirmə (Kvantlaşdırma) | |
| | Gün 34: Ollama-ya Giriş (Modelin Dağıtımı) | |
| | Gün 35: Ollama API ilə İşləmək (Chatbotun Qurulması) | |
| | Gün 36: Modelin Paylaşılması və GitHub | |
| | Gün 37: Modelin Qiymətləndirilməsi və Nəticələrin Təhlili | |
| | Gün 38: Modelin Təkmilləşdirilməsi (Iterasiya) | |
| | Gün 39: Modelin İdarə Edilməsi və Sürətləndirilməsi | |
| | Gün 40: Etik Mülahizələr və Məsuliyyətli Süni İntellekt | |
| **Hissə 5** | **Gələcək və Karyera** | Gün 41 - Gün 50 |
| | Gün 41: LLM-lərin Gələcəyi və Təkmilləşdirmə Yolları | |
| | Gün 42: Layihənin Sənədləşdirilməsi və Təqdimatı | |
| | Gün 43: Təlimin Xərcləri və Resursların İdarə Edilməsi | |
| | Gün 44: LLM-lərin Tətbiq Sahələri və Gələcək Layihələr | |
| | Gün 45: Süni İntellekt Tərtibatçısı Karyerası | |
| | Gün 46: Kitabın Dizaynı və Formatlaşdırılması | |
| | Gün 47: Kitabın Son Nəzarəti və Təhvil Verilməsi | |
| | Gün 48: Kitabın Ön Sözü və Məzmun Cədvəli | |
| | Gün 49: Yekun Söz və Təşəkkür | |
| | Gün 50: DOCX-ə Çevrilmə və Təhvil | |
