# 📚 50 Gündə Süni-İntellekt: Gün 36

## Modelin Paylaşılması və GitHub: Layihəni İctimailəşdirmək 🌐

Salam! Dünən modelimizi Ollama API vasitəsilə Python-dan idarə etməyi öyrəndik. Bu gün isə layihəmizi dostlarınızla və GitHub səhifənizdə paylaşmaq üçün lazım olan addımları öyrənəcəyik.

### 1. Niyə GitHub?

**GitHub** proqram təminatının inkişafı üçün ən böyük platformadır. O, kodunuzu saxlamağa, versiyalara nəzarət etməyə və başqaları ilə əməkdaşlıq etməyə imkan verir.

> **GitHub** — Layihənizin bütün kodunu, konfiqurasiya fayllarını və sənədlərini saxlayacağınız mərkəzi bir yerdir.

### 2. Layihənin Strukturunun Hazırlanması

Paylaşmadan əvvəl layihə qovluğumuzu səliqəyə salmalıyıq.

```
az-nano-llm/
├── config.py             # Modelin hiperparametrləri
├── model.py              # GPT modelinin tam arxitekturası
├── attention.py          # Diqqət mexanizmi
├── block.py              # Transformer bloku
├── data_loader.py        # Məlumat yükləyicisi
├── train.py              # Təlim skripti
├── load_model.py         # Təlim olunmuş modeli yükləmə skripti
├── az_chatbot.py         # Ollama API ilə chatbot
├── az_bpe_tokenizer.json # Tokenizator faylı
├── requirements.txt      # Lazım olan kitabxanaların siyahısı
├── README.md             # Layihənin təsviri (Çox vacibdir!)
└── .gitignore            # GitHub-a yüklənməməli fayllar
```

#### `requirements.txt` Faylı

Bu fayl layihəni işə salmaq üçün lazım olan bütün Python kitabxanalarını ehtiva edir.

```
# requirements.txt
torch
numpy
tokenizers
tqdm
accelerate
transformers
ollama
```

#### `.gitignore` Faylı

Bu fayl GitHub-a yüklənməməli olan böyük və ya şəxsi faylları göstərir.

```
# .gitignore
# Təlim məlumatları
*.npy
# Model çəkiləri (çox böyükdür)
*.pt
*.gguf
# Hugging Face qovluğu
az_llm_hf/
# Checkpoint qovluqları
checkpoint_*/
# PyTorch cache
__pycache__/
```

### 3. GitHub Repozitoriyasının Yaradılması

1.  **GitHub Hesabı:** Əgər yoxdursa, `github.com`-da hesab yaradın.
2.  **Yeni Repozitoriya:** `New` düyməsinə basaraq **`az-nano-llm`** adlı yeni bir repozitoriya yaradın.
3.  **Yerli Repozitoriyanın Başlanması:** Layihə qovluğunuzda (məsələn, `az-nano-llm`) terminalı açın və aşağıdakı əmrləri icra edin:

```bash
# 1. Git-i başlat
git init

# 2. Bütün faylları əlavə et
git add .

# 3. İlk dəyişikliyi yadda saxla
git commit -m "Initial commit: NanoGPT arxitekturası və təlim skriptləri"

# 4. Uzaq repozitoriyanı əlavə et (Sizin repozitoriyanızın linki)
git remote add origin https://github.com/SizinAdiniz/az-nano-llm.git

# 5. Faylları GitHub-a yüklə
git push -u origin master
```

### 4. Modelin Çəkilərinin Paylaşılması (GGUF Faylı)

`az_llm_q4km.gguf` faylı 62 MB-dır. GitHub 100 MB-dan kiçik faylları qəbul edir, lakin böyük faylları saxlamaq üçün **Git Large File Storage (LFS)** istifadə etmək daha yaxşıdır.

```bash
# 1. Git LFS-i quraşdırın (Əgər quraşdırılmayıbsa)
git lfs install

# 2. GGUF faylını izləməyə başla
git lfs track "*.gguf"

# 3. .gitattributes faylını əlavə et
git add .gitattributes

# 4. GGUF faylını əlavə et
git add az_llm_q4km.gguf

# 5. Commit və Push et
git commit -m "Add kvantlaşdırılmış GGUF model çəkiləri"
git push
```

### 5. README.md Faylı

**README.md** faylı layihənizin vizit kartıdır. O, layihənin nə olduğunu, necə quraşdırılacağını və necə istifadə ediləcəyini izah etməlidir.

**Əsas Hissələr:**
1.  **Başlıq:** Azərbaycan Nano LLM (100M Parametr)
2.  **Təsvir:** Modelin məqsədi (Azərbaycan dilində chatbot).
3.  **Arxitektura:** NanoGPT (GPT-2 əsaslı, 12 qat, 12 baş, 768 ölçü).
4.  **Quraşdırma:** `git clone`, `conda create`, `pip install -r requirements.txt` addımları.
5.  **İstifadə:** Ollama-da necə işə salınacağı.

### 💡 Günün Tapşırığı: Praktika

1.  Layihə qovluğunuzda `requirements.txt` və `.gitignore` fayllarını yaradın.
2.  GitHub repozitoriyasını yaradın və kodu ora yükləyin.
3.  `README.md` faylının təməlini yazın.

**Sabah görüşənədək!** 👋 Sabah **Modelin Qiymətləndirilməsi və Nəticələrin Təhlili** mövzusunu öyrənəcəyik.

***

**Söz Sayı:** 750 söz.
