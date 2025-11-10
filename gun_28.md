# Gün 28: Checkpoint və Modelin Saxlanması 💾

## 28.1. Checkpoint-in Funksional Əhəmiyyəti

**Checkpoint (Nəzarət Nöqtəsi)** təlim prosesinin müəyyən bir anında modelin vəziyyətinin yadda saxlanmasıdır.

**Məntiq:** LLM təlimi uzunmüddətli bir prosesdir. Checkpoint-lər təlimin uğursuzluq (elektrik kəsilməsi, proqram xətası) səbəbindən yarımçıq qalması riskini sığortalayır və modelin **ən yaxşı performans göstərdiyi** vəziyyəti saxlamağa imkan verir.

**Checkpoint-ə Daxil Edilənlər:**

| Məlumat | Məqsəd |
| :--- | :--- |
| **Modelin Çəkiləri** | Modelin öyrəndiyi bilik. |
| **Optimallaşdırıcının Vəziyyəti** | Təlimi davam etdirmək üçün lazım olan daxili dəyişənlər (məsələn, AdamW-nin momentləri). |
| **Cari Epoxa/Addım** | Təlimin hansı nöqtədən davam etdiriləcəyini göstərir. |
| **Ən Yaxşı Validasiya Loss-u** | Modelin ən yaxşı nəticə göstərdiyi vəziyyəti müəyyənləşdirmək. |

## 28.2. `accelerate` ilə Checkpoint Mexanizmi

`accelerate` kitabxanası Checkpoint mexanizmini sadələşdirir və bütün lazımi komponentləri (model, optimizer, scheduler) avtomatik olaraq idarə edir.

**Saxlama Məntiqi:**

Ən yaxşı Checkpoint, adətən **ən aşağı Validasiya Loss-una** malik olan Checkpoint-dir.

**`train_accelerate.py` Skriptinə Əlavə:**

```python
# ... (Əvvəlki kod) ...

# Təlim dövründən əvvəl
best_val_loss = float('inf')
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ... (Təlim dövrü) ...

for epoch in range(NUM_EPOCHS):
    # ... (Təlim və Validasiya) ...
    
    val_loss, val_ppl = validate(model, val_dataloader, accelerator)
    
    # 1. Ən Yaxşı Checkpoint-i Yoxlamaq
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        
        # 2. Checkpoint-i Saxlamaq
        # accelerate avtomatik olaraq model, optimizer və scheduler-i saxlayır.
        accelerator.save_state(os.path.join(CHECKPOINT_DIR, "best_model"))
        
        accelerator.print(f"Yeni ən yaxşı Validasiya Loss-u ({best_val_loss:.4f}) tapıldı. Checkpoint saxlandı.")
        
    # Hər epoxanın sonunda cari vəziyyəti saxlamaq (davam etdirmək üçün)
    accelerator.save_state(os.path.join(CHECKPOINT_DIR, f"epoch_{epoch}"))
```

## 28.3. Checkpoint-dən Bərpa

Təlimi dayandırılmış bir nöqtədən davam etdirmək üçün `accelerator.load_state()` funksiyasından istifadə olunur.

**Bərpa Məntiqi:**

1.  **`accelerator.load_state(path)`** funksiyası modelin çəkilərini, optimallaşdırıcının vəziyyətini və scheduler-in vəziyyətini yükləyir.
2.  Təlim dövrü yüklənmiş vəziyyətdən (məsələn, 5-ci epoxanın ortasından) davam edir.

**`train_accelerate.py` Skriptinə Bərpa Məntiqi:**

```python
# ... (Əvvəlki kod) ...

# Təlim dövründən əvvəl
CHECKPOINT_TO_LOAD = os.path.join(CHECKPOINT_DIR, "epoch_4") # Məsələn, 4-cü epoxadan davam etmək

if os.path.exists(CHECKPOINT_TO_LOAD):
    accelerator.load_state(CHECKPOINT_TO_LOAD)
    accelerator.print(f"Checkpoint '{CHECKPOINT_TO_LOAD}' uğurla yükləndi. Təlim davam etdirilir.")
    # Başlanğıc epoxasını təyin etmək
    start_epoch = int(CHECKPOINT_TO_LOAD.split('_')[-1]) + 1
else:
    start_epoch = 0

# Təlim dövrü
for epoch in range(start_epoch, NUM_EPOCHS):
    # ... (Təlim davam edir) ...
```

**Məntiq:** Bu mexanizm, xüsusilə bulud xidmətlərində və ya məhdud resurslu kompüterlərdə (RTX 2050) təlimin **etibarlılığını** və **davamlılığını** təmin edir.
