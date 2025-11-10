# Gün 25: Məhdud Resurslarda Təlimin Başlanması (RTX 2050 Optimizasiyası) ⚙️

## 25.1. `accelerate` Kitabxanasının Rolu

Məhdud VRAM (4GB) şəraitində LLM təlimini idarə etmək üçün **Hugging Face `accelerate`** kitabxanasından istifadə olunur. Bu kitabxana, **Mixed Precision (FP16)** və **Gradient Accumulation** kimi optimallaşdırmaları təlim dövrünə asanlıqla inteqrasiya etməyə imkan verir.

**Məntiq:** `accelerate` kitabxanası, təlim kodunu GPU-ya uyğunlaşdırmaq və optimallaşdırmaları tətbiq etmək üçün lazım olan bütün aşağı səviyyəli əməliyyatları avtomatik olaraq idarə edir.

## 25.2. Təlim Konfiqurasiyası

Məhdud VRAM üçün kritik konfiqurasiya parametrləri:

| Parametr | Dəyər | Məntiqi Əsas |
| :--- | :--- | :--- |
| **`mixed_precision`** | `"fp16"` | **VRAM-ı 50% azaltmaq** üçün 16-bit dəqiqlikdən istifadə. |
| **`gradient_accumulation_steps`** | `4` | **Effektiv Batch Size-ı artırmaq.** Hər 4 addımdan bir qradiyentləri toplayıb, sonra çəkiləri yeniləmək. |
| **`per_device_train_batch_size`** | `4` | 4GB VRAM-da modelin çəkiləri və aralıq hesablamalar üçün yer saxlamaq üçün ən kiçik Batch Size seçilir. |
| **Effektiv Batch Size** | `16` (4 * 4) | Təlimin keyfiyyəti üçün lazım olan Batch Size-ın simulyasiyası. |

## 25.3. Praktika: `accelerate` ilə Təlim

**`train_accelerate.py`**

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import get_linear_schedule_with_warmup
from accelerate import Accelerator
import os
# GPTModel sinfini Gün 17-dən, get_batch funksiyasını Gün 22-dən import edin

# 1. Konfiqurasiya
BLOCK_SIZE = 256
BATCH_SIZE = 4 # Per-device batch size
GRADIENT_ACCUMULATION_STEPS = 4
MAX_LR = 3e-4
NUM_EPOCHS = 10

# 2. Accelerator-un Başlanması
# Mixed Precision-ı təyin edirik
accelerator = Accelerator(mixed_precision='fp16', gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS)
device = accelerator.device

# 3. Məlumatın Yüklənməsi
train_data = torch.load('train_data.pt')
val_data = torch.load('val_data.pt')
train_dataset = TensorDataset(train_data)
val_dataset = TensorDataset(val_data)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 4. Model, Optimizer və Scheduler-in Qurulması
model = GPTModel(vocab_size=32000, block_size=BLOCK_SIZE, n_layer=12, n_head=12, n_embd=768)
optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR)

# Təlim addımlarının ümumi sayı
total_steps = len(train_dataloader) * NUM_EPOCHS // GRADIENT_ACCUMULATION_STEPS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps)

# 5. Modelin Accelerator-a Hazırlanması
# Bu, model, optimizer və dataloader-i avtomatik olaraq GPU-ya köçürür və FP16-ya uyğunlaşdırır.
model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, val_dataloader, scheduler
)

# 6. Təlim Dövrü
for epoch in range(NUM_EPOCHS):
    model.train()
    for step, batch in enumerate(train_dataloader):
        # Batch-i giriş və hədəfə ayırmaq
        X, Y = batch[0][:, :-1], batch[0][:, 1:] 
        
        # Qradiyent yığımı üçün kontekst meneceri
        with accelerator.accumulate(model):
            logits, loss = model(X, Y)
            
            # Loss-u geri yaymaq (Backpropagation)
            accelerator.backward(loss)
            
            # Qradiyentləri kəsmək (Gradient Clipping)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            
            # Optimizer addımı və Scheduler yenilənməsi
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
        if step % 100 == 0:
            accelerator.print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")
            
    # Hər epoxanın sonunda Validasiya və Checkpoint saxlamaq
    # ... (Gün 27 və Gün 28-də öyrəniləcək) ...

# 7. Təlimin Başlanması
# Terminalda icra:
# accelerate launch train_accelerate.py
```

## 25.4. Kodun Məntiqi İzahı

| Sətr | Kod | Məntiqi Əsas |
| :--- | :--- | :--- |
| **19** | `accelerator = Accelerator(mixed_precision='fp16', ...)` | **Kritik:** `accelerator` obyektini yaradırıq. Bu, bütün optimallaşdırmaların mərkəzi idarəetmə nöqtəsidir. `mixed_precision='fp16'` 4GB VRAM üçün həyati əhəmiyyət kəsb edir. |
| **38** | `model, optimizer, ... = accelerator.prepare(...)` | **Əsas Addım:** Model, optimizer və dataloader-i `accelerator` tərəfindən idarə olunmaq üçün hazırlayır. Bu, avtomatik olaraq GPU-ya köçürməni və FP16-ya çevirməni həyata keçirir. |
| **45** | `with accelerator.accumulate(model):` | **Gradient Accumulation Məntiqi:** Bu kontekst meneceri, `GRADIENT_ACCUMULATION_STEPS` dəfə `accelerator.backward(loss)` çağırılana qədər qradiyentləri yığır və yalnız sonra `optimizer.step()` əməliyyatına icazə verir. |
| **47** | `accelerator.backward(loss)` | **FP16 Uyğunluğu:** PyTorch-un standart `loss.backward()` əvəzinə `accelerator.backward()` istifadə olunur. Bu, FP16-da təhlükəsiz və düzgün geri yayılmanı təmin edir. |
| **50** | `accelerator.clip_grad_norm_(model.parameters(), 1.0)` | **Qradiyentləri Kəsmək (Clipping):** Qradiyentlərin həddindən artıq böyüməsinin (Exploding Gradients) qarşısını alır. Bu, LLM təlimində sabitliyi təmin edən vacib bir texnikadır. |
| **57** | `accelerate launch train_accelerate.py` | **İcra:** Skript PyTorch-un standart `python train.py` əmri ilə deyil, `accelerate launch` əmri ilə işə salınmalıdır. |
