# 📚 50 Gündə Süni-İntellekt: Gün 23

## Təlim Dövrü (Training Loop): Modelin Öyrənmə Prosesi 🔄

Salam! Dünən modelimizi məlumatla təchiz edəcək **DataLoader**-i qurduq. Bu gün isə bütün komponentləri birləşdirərək **Təlim Dövrünü (Training Loop)** – yəni modelin əsl öyrənmə prosesini – yazırıq.

Bu, LLM layihəmizin **ən vacib** kod hissəsidir.

### 1. Təlim Dövrünün Əsas Addımları

Təlim dövrü hər bir **Batch** (məlumat dəsti) üçün ardıcıl olaraq dörd əsas addımı təkrarlayır:

1.  **Məlumatın Yüklənməsi:** DataLoader-dən bir Batch (giriş `x` və hədəf `y`) alınır.
2.  **İrəli Ötürmə (Forward Pass):** Giriş `x` modelə verilir və çıxış `logits` və `loss` hesablanır.
3.  **Geriyə Ötürmə (Backward Pass):** `loss.backward()` əmri ilə qradiyentlər hesablanır.
4.  **Parametrlərin Yenilənməsi:** Optimallaşdırıcı (AdamW) qradiyentləri istifadə edərək modelin çəkilərini tənzimləyir.

### 2. Təlim Skriptinin Tamamlanması

İndi `train.py` skriptini tamamlayırıq.

```python
# train.py
import torch
from torch.utils.data import Dataset, DataLoader
from config import GPTConfig
from model import GPT
from data_loader import get_dataloaders
from accelerate import Accelerator
from tqdm import tqdm
import time

# 1. Hiperparametrlər
BATCH_SIZE = 12
BLOCK_SIZE = 512
LEARNING_RATE = 6e-4
MAX_ITERS = 5000
EVAL_INTERVAL = 500
EVAL_ITERS = 200 # Validasiya üçün istifadə olunan Batch sayı
GRADIENT_ACCUMULATION_STEPS = 4 # Qradiyent yığımı üçün addım sayı

# 2. Akseleratoru Başlatmaq (Mixed Precision üçün)
accelerator = Accelerator(
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    mixed_precision='fp16' # Yaddaşa qənaət edən 16-bit dəqiqlik
)
device = accelerator.device

# 3. Model, DataLoader və Optimizer-i Hazırlamaq
config = GPTConfig(block_size=BLOCK_SIZE)
model = GPT(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
train_loader, val_loader = get_dataloaders(BLOCK_SIZE, BATCH_SIZE)

# Akselerator ilə bütün obyektləri GPU-ya köçürürük
model, optimizer, train_loader, val_loader = accelerator.prepare(
    model, optimizer, train_loader, val_loader
)

# 4. Validasiya Funksiyası
@torch.no_grad()
def estimate_loss():
    """ Validasiya məlumatı üzərində itkini hesablayır """
    model.eval() # Modeli qiymətləndirmə rejiminə keçiririk
    losses = []
    for _ in range(EVAL_ITERS):
        # Validasiya Batch-ini yüklə
        x, y = next(iter(val_loader))
        # İrəli ötürmə
        with accelerator.autocast():
            logits, loss = model(x, targets=y)
        losses.append(accelerator.gather(loss).mean().item())
    
    model.train() # Modeli təlim rejiminə qaytarırıq
    return torch.tensor(losses).mean().item()

# 5. Əsas Təlim Dövrü
start_time = time.time()
for iter_num in tqdm(range(MAX_ITERS), desc="Təlim Prosesi"):
    
    # A. Validasiya
    if iter_num % EVAL_INTERVAL == 0:
        val_loss = estimate_loss()
        print(f"Addım {iter_num}: Təlim İtkisi (Loss) = {val_loss:.4f}")
        # Modelin vəziyyətini yadda saxlamaq (Checkpoint)
        # accelerator.save_state(f"checkpoint_{iter_num}")

    # B. Məlumatı Yüklə
    x, y = next(iter(train_loader))
    
    # C. İrəli Ötürmə və İtki Hesablanması
    with accelerator.accumulate(model):
        with accelerator.autocast(): # Mixed Precision üçün
            logits, loss = model(x, targets=y)
        
        # D. Geriyə Ötürmə və Parametrlərin Yenilənməsi
        # Qradiyent yığımı (accumulation) ilə birlikdə geriyə ötürmə
        accelerator.backward(loss)
        
        # Qradiyentlərin kəsilməsi (Gradient Clipping) - partlayan qradiyentlərin qarşısını alır
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            
        optimizer.step() # Parametrləri yenilə
        optimizer.zero_grad() # Qradiyentləri sıfırla

end_time = time.time()
print(f"\nTəlim tamamlandı. Ümumi vaxt: {(end_time - start_time) / 3600:.2f} saat")
```

### 3. Kodun İzahı (Hər Sətrin Detallı İzahı)

| Sətr | Kod | İzah |
| :--- | :--- | :--- |
| 17 | `GRADIENT_ACCUMULATION_STEPS = 4` | **Qradiyent Yığımı:** Hər 4 Batch-dən bir parametrləri yeniləyəcəyik. Bu, **effektiv Batch Size-ı** 4 dəfə artırır (12 * 4 = 48). |
| 21 | `accelerator = Accelerator(...)` | **Akselerator:** GPU-nu, `fp16` (16-bit dəqiqlik) və qradiyent yığımını idarə edən əsas obyektdir. |
| 32 | `model, optimizer, ... = accelerator.prepare(...)` | Bütün PyTorch obyektlərini avtomatik olaraq GPU-ya köçürür və `fp16` üçün hazırlayır. |
| 37 | `@torch.no_grad()` | Validasiya zamanı qradiyent hesablanmasını söndürür. |
| 38 | `model.eval()` | Modeli qiymətləndirmə rejiminə keçirir (Dropout və LayerNorm fərqli işləyir). |
| 58 | `if iter_num % EVAL_INTERVAL == 0:` | Hər 500 addımdan bir validasiya itkisini hesablayıb ekrana çıxarır. |
| 66 | `with accelerator.accumulate(model):` | Bu blokun içindəki `backward` əmri, `GRADIENT_ACCUMULATION_STEPS` qədər qradiyentləri yığacaq. |
| 67 | `with accelerator.autocast():` | **Mixed Precision** (Qarışıq Dəqiqlik) tətbiq edir. Bəzi əməliyyatlar `fp16` ilə, bəziləri isə `fp32` ilə icra olunur. Bu, yaddaşa qənaət edir. |
| 70 | `accelerator.backward(loss)` | **Geriyə Ötürmə** əmri. |
| 73 | `accelerator.clip_grad_norm_(model.parameters(), 1.0)` | **Qradiyent Kəsilməsi:** Qradiyentlərin dəyərini 1.0-dan yuxarı qalxmasının qarşısını alır. Bu, təlimin stabil qalması üçün vacibdir. |
| 75 | `optimizer.step()` | Yığılmış qradiyentlərə əsasən modelin çəkilərini yeniləyir. |
| 76 | `optimizer.zero_grad()` | Növbəti Batch üçün qradiyentləri sıfırlayır. |

### 💡 Günün Tapşırığı: Praktika

1.  **`train.py`** faylını tamamlayın.
2.  Bütün asılılıqların (model.py, data_loader.py, config.py, az_bpe_tokenizer.json, train.npy, val.npy) hazır olduğundan əmin olun.
3.  **Təlimə başlayın!** `accelerate launch train.py` əmri ilə təlimi başladın.

**Sabah görüşənədək!** 👋 Sabah **Optimallaşdırıcı və Öyrənmə Sürəti** mövzusunu daha dərindən araşdıracağıq.

***

**Söz Sayı:** 850 söz.
