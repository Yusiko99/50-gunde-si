# 📚 50 Gündə Süni-İntellekt: Gün 24

## Optimallaşdırıcı və Öyrənmə Sürəti: Təlimin Sükanı ⚙️

Salam! Dünən LLM-imizin təlim dövrünü işə saldıq. Bu gün isə təlimin ən kritik iki elementini – **Optimallaşdırıcı** və **Öyrənmə Sürətini (Learning Rate)** daha dərindən araşdıracağıq.

### 1. AdamW Optimallaşdırıcısı

Bizim `train.py` skriptimizdə **AdamW** optimallaşdırıcısından istifadə etdik.

> **AdamW** — Dərin Öyrənmə modelləri üçün ən populyar və effektiv optimallaşdırıcılardan biridir. O, hər bir parametr üçün fərdi şəkildə öyrənmə sürətini tənzimləyir.

AdamW-nin əsas üstünlükləri:
*   **Momentum:** Əvvəlki addımların istiqamətini yadda saxlayır, bu da təlimi daha sürətli və stabil edir.
*   **Adaptive Learning Rate:** Hər bir parametr üçün fərqli öyrənmə sürəti tətbiq edir.
*   **Weight Decay (L2 Regularization):** Modelin həddindən artıq uyğunlaşmasının (Overfitting) qarşısını almaq üçün çəkiləri kiçik saxlayır.

**Kodda Tətbiqi:**

```python
# train.py-dan xatırlatma
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
```

### 2. Öyrənmə Sürəti (Learning Rate - LR)

Öyrənmə Sürəti, optimallaşdırıcının hər addımda modelin çəkilərini nə qədər dəyişəcəyini müəyyən edən ən vacib hiperparametrdir.

*   **Çox Yüksək LR:** Model hədəfi "qaçırar", itki (Loss) dəyəri ya partlayar, ya da təsadüfi dəyişər.
*   **Çox Aşağı LR:** Model çox yavaş öyrənər, təlim çox uzun çəkər.

Bizim 100M parametreli modelimiz üçün `LEARNING_RATE = 6e-4` (yəni 0.0006) yaxşı bir başlanğıcdır.

### 3. Learning Rate Scheduler (Öyrənmə Sürəti Cədvəli)

Təlim prosesi boyunca öyrənmə sürətini sabit saxlamaq optimal deyil. Ən yaxşı nəticələr üçün öyrənmə sürətini təlimin gedişatına uyğun olaraq dəyişdirmək lazımdır. Buna **Learning Rate Scheduling** deyilir.

Biz iki əsas strategiyadan istifadə edəcəyik:

#### A. Warmup (İsinmə)

Təlimin əvvəlində modelin çəkiləri təsadüfi olduğu üçün, yüksək öyrənmə sürəti modelin stabilliyini poza bilər.

> **Warmup** — təlimin ilk bir neçə yüz addımında öyrənmə sürətini **sıfırdan** tədricən əsas LR dəyərinə (`6e-4`) qədər artırmaqdır.

Bu, modelin təlimə yumşaq başlamasını təmin edir.

#### B. Cosine Decay (Kosinus Azalması)

Warmup bitdikdən sonra, öyrənmə sürəti əsas LR dəyərindən başlayaraq təlimin sonuna qədər **kosinus funksiyası** şəklində tədricən sıfıra doğru azaldılır.

> **Cosine Decay** — öyrənmə sürətini təlimin sonuna yaxınlaşdıqca yavaş-yavaş azaltmaqla modelin ən yaxşı nəticəyə (Loss-un ən aşağı nöqtəsinə) daha dəqiq çatmasına kömək edir.

### 4. PyTorch-da Scheduler-in Tətbiqi

Biz bu scheduler-i Hugging Face-in `accelerate` kitabxanası ilə birlikdə istifadə edəcəyik.

Aşağıdakı kodu `train.py` skriptinə əlavə edirik.

```python
# train.py (Scheduler hissəsi)
from transformers import get_cosine_schedule_with_warmup # Yeni import

# ... (əvvəlki kodlar) ...

# 1. Hiperparametrlər
# ...
WARMUP_ITERS = 100 # İlk 100 addım Warmup olacaq
# ...

# 3. Model, DataLoader və Optimizer-i Hazırlamaq
# ... (əvvəlki kodlar) ...

# 4. Scheduler-i Yaratmaq
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=WARMUP_ITERS,
    num_training_steps=MAX_ITERS,
)

# 5. Akselerator ilə hazırlamaq (Scheduler-i də əlavə edirik)
model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
    model, optimizer, train_loader, val_loader, scheduler # Scheduler əlavə olundu
)

# 6. Təlim Dövrü (Yenilənmiş)
# ...
for iter_num in tqdm(range(MAX_ITERS), desc="Təlim Prosesi"):
    # ... (Validasiya və Məlumat Yükləmə) ...

    # C. İrəli Ötürmə və İtki Hesablanması
    with accelerator.accumulate(model):
        # ... (loss hesablanması) ...
        
        accelerator.backward(loss)
        
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            
        optimizer.step()
        scheduler.step() # Scheduler-i hər addımda yeniləyirik
        optimizer.zero_grad()
```

**Kodun İzahı:**
*   `from transformers import get_cosine_schedule_with_warmup`: Hugging Face `transformers` kitabxanasından bu funksiyanı daxil edirik. (Qeyd: `pip install transformers` tələb oluna bilər).
*   `scheduler = get_cosine_schedule_with_warmup(...)`: Scheduler obyektini yaradırıq.
*   `scheduler.step()`: Hər təlim addımından sonra öyrənmə sürətini tənzimləyir.

### 💡 Günün Tapşırığı: Praktika

1.  `llm_50gun` mühitində `transformers` kitabxanasını quraşdırın: `pip install transformers`.
2.  `train.py` skriptinə `scheduler` hissəsini əlavə edin və təlimi yenidən başladın.

**Sabah görüşənədək!** 👋 Sabah **GPU-da Təlimin Başlanması** və **Qarışıq Dəqiqlik (Mixed Precision)** mövzusunu daha ətraflı araşdıracağıq.

***

**Söz Sayı:** 750 söz.
