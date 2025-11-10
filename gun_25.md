# Gün 25: RTX 2050-də Təlimin Başlanması (Optimallaşdırma) 🚀

## 25.1. Niyə `accelerate`?

Əvvəlki günlərdə biz **NanoGPT** modelini və təlim dövrünü PyTorch-da qurduq. İndi isə bu təlim dövrünü sizin **RTX 2050 (4GB VRAM)** kartınız üçün optimallaşdırmalıyıq.

**`accelerate`** kitabxanası Hugging Face tərəfindən yaradılmışdır və bizə **Distributed Training (Paylanmış Təlim)**, **Mixed Precision (FP16)** və **Gradient Accumulation** kimi mürəkkəb optimallaşdırmaları **sadəcə bir neçə sətir kodla** tətbiq etməyə imkan verir.

## 25.2. `accelerate` ilə Təlim Dövrünün Hazırlanması

Bizim `train.py` skriptimizdə dəyişikliklər edərək `accelerate` istifadə edəcəyik.

**`train_accelerate.py` (Əsas dəyişikliklər)**

```python
# ... (Əvvəlki importlar və model/data yüklənməsi) ...
from accelerate import Accelerator

# 1. Accelerator-un yaradılması
# Mixed Precision-ı avtomatik tətbiq edəcək
accelerator = Accelerator(
    gradient_accumulation_steps=4, # Gradient Accumulation addımı
    mixed_precision='fp16' # RTX 2050 üçün kritik optimallaşdırma
)

# 2. Model, Optimallaşdırıcı və DataLoader-in Accelerator-a ötürülməsi
model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, val_dataloader
)

# ... (Təlim dövrü) ...

# 3. Gradient Accumulation-ın tətbiqi
for step, batch in enumerate(train_dataloader):
    with accelerator.accumulate(model):
        # ... (Forward pass və loss hesablanması) ...
        
        # Loss-u geri yaymaq (Backpropagation)
        accelerator.backward(loss)
        
        # Qradiyentləri yeniləmək
        optimizer.step()
        optimizer.zero_grad()
        
    # ... (Monitorinq və Checkpoint) ...
```

## 25.3. RTX 2050 üçün Kritik Parametrlər

Sizin 4GB VRAM-ınız üçün ən vacib konfiqurasiya addımları bunlardır:

### A. Mixed Precision (FP16)

`accelerator = Accelerator(mixed_precision='fp16')` əmri modelin çəkilərini və əməliyyatlarını 16-bit dəqiqlikdə aparmağa məcbur edir. Bu, **VRAM istifadəsini təxminən 50% azaldır**.

### B. Gradient Accumulation (Qradiyent Yığımı)

`gradient_accumulation_steps=4` təyin etdik.

*   **Mini Batch Size (Həqiqi Batch Size):** Tutaq ki, VRAM-ınız yalnız **Batch Size = 4**-ə icazə verir.
*   **Gradient Accumulation Steps:** 4
*   **Effektiv Batch Size:** $4 \times 4 = 16$

Bu o deməkdir ki, model hər 4 kiçik Batch-dən sonra bir dəfə çəkilərini yeniləyəcək. Bu, 4GB VRAM-da belə, daha böyük Batch Size-ın təsirini simulyasiya etməyə imkan verir.

## 25.4. Təlimin Başlanması

Təlimi başlatmaq üçün sadəcə `python train.py` əvəzinə `accelerate` istifadə edirik:

**Addım 1: Konfiqurasiya Faylının Yaradılması**

Terminalda `accelerate config` əmrini icra edin. Bu, kitabxananın sizin sisteminizi (GPU, VRAM) tanıyıb uyğun parametrləri təyin etməsinə kömək edir.

**Əsas Konfiqurasiya Seçimləri:**

| Sual | Cavab (RTX 2050 üçün) | İzahı |
| :--- | :--- | :--- |
| **How many GPUs are you using?** | 1 | Tək GPU istifadə edirik. |
| **Do you wish to use FP16 or BF16?** | **fp16** | **Kritik:** VRAM-ı 50% azaltmaq üçün FP16-nı seçirik. |
| **Do you want to use DeepSpeed?** | No | DeepSpeed daha böyük modellər üçündür. |

**Addım 2: Təlimin Başlanması**

```bash
accelerate launch train_accelerate.py
```

Bu əmr `accelerate` konfiqurasiyanızı oxuyacaq, FP16 və Gradient Accumulation-ı tətbiq edəcək və təlimi optimallaşdırılmış şəkildə başladacaq.

**Gündəlik Tapşırıq:** `train_accelerate.py` skriptini Gün 23-dəki `train.py` skriptinə əsaslanaraq yeniləyin. Terminalda `accelerate config` əmrini icra edin və konfiqurasiya faylını yaradın.
