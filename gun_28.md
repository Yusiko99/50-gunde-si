# Gün 28: Checkpoint və Modelin Saxlanması 💾

## 28.1. Checkpoint Nədir?

**Checkpoint (Nəzarət Nöqtəsi)** təlim prosesinin müəyyən bir anında modelin vəziyyətinin (çəkilərinin, optimallaşdırıcının vəziyyətinin, cari epoxanın) yadda saxlanmasıdır.

**Niyə Checkpoint Vacibdir?**

1.  **Fasiləsiz Təlim:** Təlim prosesi elektrik kəsilməsi, proqram xətası və ya serverin bağlanması səbəbindən dayandırılarsa, son Checkpoint-dən davam etmək mümkündür. Bu, vaxta və resurslara qənaət edir.
2.  **Modelin Təhlili:** Təlimin müxtəlif mərhələlərindəki modelləri (məsələn, 1-ci epoxa, 5-ci epoxa) saxlamaq və sonradan müqayisə etmək.

## 28.2. PyTorch-da Checkpoint Saxlanması

PyTorch-da Checkpoint saxlamaq üçün adətən bir lüğət (dictionary) istifadə olunur. Bu lüğətə modelin çəkiləri ilə yanaşı, təlimin davam etdirilməsi üçün lazım olan bütün məlumatlar daxil edilir.

**Checkpoint-ə Daxil Edilənlər:**

| Məlumat | Məqsəd |
| :--- | :--- |
| **`model.state_dict()`** | Modelin bütün öyrənilmiş çəkiləri. |
| **`optimizer.state_dict()`** | Optimallaşdırıcının cari vəziyyəti (məsələn, AdamW-nin daxili dəyişənləri). |
| **`epoch` / `step`** | Təlimin hansı mərhələdə dayandığını göstərir. |
| **`loss`** | Cari və ya ən yaxşı Validasiya Loss-u. |

**Checkpoint Saxlama Funksiyası:**

```python
def save_checkpoint(model, optimizer, epoch, loss, path):
    """Modelin vəziyyətini yadda saxlayır."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint '{path}' faylına uğurla yazıldı.")
```

## 28.3. Checkpoint-dən Bərpa

Təlimi Checkpoint-dən bərpa etmək üçün:

```python
def load_checkpoint(model, optimizer, path):
    """Modelin vəziyyətini Checkpoint-dən bərpa edir."""
    if not os.path.exists(path):
        print(f"Xəta: Checkpoint faylı '{path}' tapılmadı.")
        return 0 # 0-cı epoxadan başla
        
    checkpoint = torch.load(path)
    
    # Model və Optimallaşdırıcının çəkilərini bərpa et
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"Checkpoint-dən bərpa olundu. Epoch: {epoch}, Loss: {loss:.4f}")
    return epoch + 1 # Növbəti epoxadan davam et
```

## 28.4. `accelerate` ilə Checkpoint

Əgər siz Gün 25-də öyrəndiyimiz kimi `accelerate` istifadə edirsinizsə, proses daha da sadələşir:

```python
# Saxlamaq
accelerator.save_state("checkpoint_dir")

# Bərpa etmək
accelerator.load_state("checkpoint_dir")
```

`accelerate` avtomatik olaraq modelin, optimallaşdırıcının vəziyyətini və digər lazım olan bütün məlumatları yadda saxlayır və bərpa edir. **RTX 2050** kimi məhdud resurslu cihazlarda təlim edərkən, **`accelerate` ilə Checkpoint** istifadə etmək ən tövsiyə olunan yoldur.

**Gündəlik Tapşırıq:** `train_accelerate.py` skriptinizə `accelerator.save_state()` əmrini əlavə edin. Məsələn, hər epoxanın sonunda və ya Validasiya Loss-u ən yaxşı olduğu zaman.
