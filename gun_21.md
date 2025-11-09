# 📚 50 Gündə Süni-İntellekt: Gün 21

## Təlim Prosesinə Giriş: Model Necə Öyrənir? 🎓

Salam! İkinci mərhələni tamamladıq və **100 Milyon parametreli NanoGPT** modelimizi PyTorch-da qurduq. İndi isə ən həyəcanlı mərhələyə – **Modelin Təliminə** başlayırıq!

Təlim prosesi, modelin Azərbaycan dilini öyrənməsi üçün rəqəmləşdirilmiş `train.npy` məlumatımızı istifadə etməsi deməkdir.

### 1. Təlim Nədir?

Təlim, modelin proqnozları ilə həqiqi dəyərlər arasındakı fərqi (İtki, yəni Loss) minimuma endirmək üçün modelin **parametr çəkilərini** (weights) tədricən tənzimləmək prosesidir.

Bu proses üç əsas komponentdən ibarətdir:

1.  **Loss Function (İtki Funksiyası):** Modelin nə qədər səhv etdiyini ölçür.
2.  **Optimizer (Optimallaşdırıcı):** İtkini azaltmaq üçün parametrləri hansı istiqamətdə və nə qədər dəyişəcəyini müəyyən edir.
3.  **Backpropagation (Geriyə Ötürmə):** İtkinin modelin bütün qatlarına necə paylandığını hesablayır.

### 2. Loss Function (İtki Funksiyası)

Bizim modelimiz **növbəti tokeni proqnozlaşdırmaq** üçün təlim olunur.

Məsələn, modelə "Azərbaycanın paytaxtı" verilir. Model proqnozlaşdırmalıdır ki, növbəti token "Bakı" sözünün token ID-si olmalıdır.

Modelin çıxışı **32000** ehtimaldan ibarət bir vektordur (hər token üçün bir ehtimal).

> **Cross-Entropy Loss (Çarpaz Entropiya İtkisi)** — generativ dil modelləri üçün standart itki funksiyasıdır. O, modelin proqnozlaşdırdığı ehtimallar paylanması ilə həqiqi tokenin paylanması arasındakı fərqi ölçür.

*   **Yüksək Loss:** Model səhv proqnozlaşdırıb.
*   **Aşağı Loss:** Model düzgün proqnozlaşdırıb.

Bizim `model.py` faylındakı `forward` metodunda bu itkini artıq hesablamışdıq:

```python
# model.py-dan xatırlatma
# ...
loss = F.cross_entropy(logits, targets)
# ...
```

### 3. Optimizer (Optimallaşdırıcı)

İtki funksiyası modelin nə qədər səhv etdiyini deyir, lakin **Optimallaşdırıcı** bu səhvi düzəltmək üçün nə etməli olduğunu deyir.

Optimallaşdırıcı, **Qradiyent Enişi (Gradient Descent)** adlı bir alqoritmə əsaslanır.

> **Qradiyent Enişi** — İtki funksiyasının qrafikində ən aşağı nöqtəni (ən yaxşı proqnozları) tapmaq üçün parametrləri qradiyentin (törəmənin) əks istiqamətində kiçik addımlarla hərəkət etdirən riyazi üsuldur.

Bizim layihəmizdə ən müasir və effektiv optimallaşdırıcılardan biri olan **AdamW**-dən istifadə edəcəyik.

> **AdamW** — **Adam** optimallaşdırıcısının **Weight Decay** (Çəki Azalması) mexanizmi ilə təkmilləşdirilmiş versiyasıdır. Weight Decay, modelin həddindən artıq uyğunlaşmasının (Overfitting) qarşısını almaq üçün parametrləri kiçik saxlayır.

### 4. Təlim üçün Əsas Kitabxanalar

Təlim prosesini idarə etmək üçün əlavə kitabxanalar quraşdırmalıyıq:

1.  **`accelerate` (Hugging Face):** Təlimi avtomatik olaraq GPU-ya (və ya bir neçə GPU-ya) uyğunlaşdırmaq üçün istifadə olunur. Bu, bizim **Mixed Precision** (Qarışıq Dəqiqlik) təlimini asanlıqla tətbiq etməyimizə kömək edəcək.
2.  **`tiktoken` (OpenAI):** Bəzi GPT-lər üçün tokenizatorları idarə etmək üçün istifadə olunur (bizim BPE tokenizatorumuz üçün birbaşa lazım olmasa da, GPT layihələrində standartdır).

#### Quraşdırma

`llm_50gun` mühitində quraşdıraq:

```bash
pip install accelerate tiktoken
```

### 5. Təlim Skriptinin Təməli

Sabah **DataLoader**-i quracağıq. Bu gün isə gələcək **`train.py`** skriptimizin təməlini qoyaq.

```python
# train.py (Təməl)
import torch
from torch.utils.data import Dataset, DataLoader
from config import GPTConfig
from model import GPT
from accelerate import Accelerator
from tqdm import tqdm

# 1. Hiperparametrlər
BATCH_SIZE = 12 # Eyni anda emal olunan cümlə sayı
BLOCK_SIZE = 512 # Cümlənin maksimum uzunluğu
LEARNING_RATE = 6e-4 # Öyrənmə sürəti (Çox vacib parametr!)
MAX_ITERS = 5000 # Maksimum təlim addımı sayı
EVAL_INTERVAL = 500 # Hər 500 addımdan bir validasiya etmək

# 2. Akseleratoru Başlatmaq
# Bu, GPU və Mixed Precision-ı idarə edəcək
accelerator = Accelerator()
device = accelerator.device

# 3. Model və Optimallaşdırıcını Yaratmaq
config = GPTConfig(block_size=BLOCK_SIZE)
model = GPT(config)
model.to(device)

# Optimallaşdırıcı
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# 4. Təlim Dövrü (Sabah tamamlanacaq)
# for iter in tqdm(range(MAX_ITERS), desc="Təlim"):
#     # 1. Məlumatı yüklə
#     # 2. İtkini hesablayıb geriyə ötür
#     # 3. Parametrləri yenilə
#     pass
```

### 💡 Günün Tapşırığı: Praktika

1.  `llm_50gun` mühitində `accelerate` və `tiktoken` kitabxanalarını quraşdırın.
2.  **`train.py`** faylını yaradın və yuxarıdakı təməl kodu ora kopyalayın.

**Sabah görüşənədək!** 👋 Sabah **Verilənlər Yükləyicisi (DataLoader)** sinfini quracağıq.

***

**Söz Sayı:** 750 söz.
