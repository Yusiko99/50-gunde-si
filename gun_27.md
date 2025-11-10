# Gün 27: Validasiya və Qiymətləndirmə (Overfitting-in Qarşısının Alınması) 🛡️

## 27.1. Validasiyanın Məntiqi Əsası

**Validasiya (Validation)** prosesi, modelin təlim məlumatları üzərində deyil, **görmədiyi** (Validasiya) məlumatlar üzərindəki performansını ölçmək üçün istifadə olunur.

**Məntiq:** Təlim Loss-unun azalması modelin öyrəndiyini göstərir, lakin Validasiya Loss-unun azalması modelin **ümumiləşdirmə (generalization)** qabiliyyətini göstərir.

| Vəziyyət | Təlim Loss-u | Validasiya Loss-u | Nəticə |
| :--- | :--- | :--- | :--- |
| **Normal Təlim** | Azalır | Azalır | Model həm öyrənir, həm də ümumiləşdirir. |
| **Overfitting** | Azalır | Artır | Model təlim məlumatlarını **əzbərləyir**, lakin yeni məlumatları proqnozlaşdıra bilmir. **Təlimi dayandırmaq lazımdır.** |
| **Underfitting** | Yüksək | Yüksək | Model kifayət qədər öyrənməyib. Daha uzun təlim və ya daha böyük model tələb olunur. |

## 27.2. Praktika: Validasiya Funksiyası

Validasiya, təlim dövründən kənarda, adətən hər epoxanın sonunda icra olunur.

**`train_accelerate.py` Skriptinə Əlavə:**

```python
@torch.no_grad() # Qradiyent hesablamasını söndürmək
def validate(model, val_dataloader, accelerator):
    """Validasiya məlumatları üzərində modelin performansını ölçür."""
    model.eval() # Modeli proqnozlaşdırma rejiminə keçirmək
    total_loss = 0
    
    # Validasiya dataloader-i üzərində iterasiya
    for batch in val_dataloader:
        X, Y = batch[0][:, :-1], batch[0][:, 1:]
        
        # Modelin proqnozlaşdırılması
        # Loss hesablamaq üçün modelin çıxışını istifadə edirik
        logits, loss = model(X, Y)
        
        # Loss-u toplamaq
        total_loss += loss.item()
        
    avg_loss = total_loss / len(val_dataloader)
    
    # Perplexity (PPL) hesablamaq
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    
    model.train() # Modeli təlim rejiminə qaytarmaq
    
    return avg_loss, ppl

# ... (Təlim dövrü) ...

for epoch in range(NUM_EPOCHS):
    # ... (Təlim addımları) ...
    
    # Hər epoxanın sonunda Validasiya
    val_loss, val_ppl = validate(model, val_dataloader, accelerator)
    
    accelerator.print(f"--- Epoch {epoch} Validasiya Nəticələri ---")
    accelerator.print(f"Validasiya Loss: {val_loss:.4f}")
    accelerator.print(f"Validasiya Perplexity (PPL): {val_ppl:.2f}")
    
    # TensorBoard-a loglamaq
    writer.add_scalar('Loss/Validation', val_loss, global_step)
    writer.add_scalar('Perplexity/Validation', val_ppl, global_step)
    
    # Checkpoint saxlamaq (Gün 28-də öyrəniləcək)
    # Ən yaxşı Validasiya Loss-u olan modeli saxlamaq lazımdır.
```

## 27.3. Kodun Məntiqi İzahı

| Sətr | Kod | Məntiqi Əsas |
| :--- | :--- | :--- |
| **1** | `@torch.no_grad()` | **Kritik:** Validasiya zamanı qradiyentlərin hesablanmasına ehtiyac yoxdur. Bu, həm hesablama sürətini artırır, həm də VRAM istifadəsini azaldır. |
| **2** | `model.eval()` | **Məntiq:** Modeli **Evaluation (Qiymətləndirmə)** rejiminə keçirir. Bu, **Dropout** və **Batch Normalization** kimi təlimə xas olan mexanizmləri söndürür. |
| **14** | `ppl = torch.exp(torch.tensor(avg_loss)).item()` | **Perplexity Hesablanması:** Loss-un eksponensial funksiyasıdır. Bu, modelin dil üzərindəki qabiliyyətini daha asan başa düşülən bir ölçü ilə ifadə edir. |
| **16** | `model.train()` | Validasiya bitdikdən sonra modelin təlim rejiminə qaytarılması vacibdir. |
| **24** | `val_loss, val_ppl = validate(model, val_dataloader, accelerator)` | **Overfitting-in Aşkarlanması:** Təlim Loss-u azalarkən Validasiya Loss-u artmağa başlasa, bu, Overfitting-in başlanğıcıdır və təlim dayandırılmalıdır. |
