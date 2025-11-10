# Gün 26: Təlimin Monitorinqi 📊

## 26.1. Niyə Monitorinq Vacibdir?

Modelin təlimi uzun və resurs tələb edən bir prosesdir. Təlimin gedişatını izləmək (monitorinq) aşağıdakılar üçün vacibdir:

1.  **Erkən Xəbərdarlıq:** Modelin öyrənmədiyini (Loss-un azalmaması) və ya həddindən artıq öyrəndiyini (Overfitting) erkən aşkar etmək.
2.  **Resurs İdarəetməsi:** GPU-nun VRAM istifadəsini və temperaturunu izləmək.
3.  **Qərar Qəbulu:** Təlimi nə vaxt dayandırmaq lazım olduğunu müəyyənləşdirmək.

Biz təlimi izləmək üçün **Loss (İtki)** və **Perplexity (PPL)** metrikalarından istifadə edəcəyik.

## 26.2. Əsas Metrikalar

### A. Loss (İtki)

**Loss** modelin proqnozları ilə həqiqi nəticələr arasındakı fərqi göstərən rəqəmdir.

*   **Təlim Loss-u (Training Loss):** Modelin təlim məlumatları üzərində nə qədər yaxşı işlədiyini göstərir.
*   **Validasiya Loss-u (Validation Loss):** Modelin **görmədiyi** məlumatlar üzərində nə qədər yaxşı ümumiləşdirdiyini göstərir.

**İdeal Senari:** Həm Təlim, həm də Validasiya Loss-u zamanla azalmalıdır.

### B. Perplexity (PPL)

**Perplexity** (Çətinlik/Qeyri-müəyyənlik) dil modellərinin keyfiyyətini ölçmək üçün istifadə olunan daha intuitiv bir metrikadır.

*   **İzahı:** Modelin növbəti tokeni proqnozlaşdırmaqda nə qədər "çaşqın" olduğunu göstərir.
*   **Dəyər:** Loss-un eksponensialı kimi hesablanır: $PPL = e^{\text{Loss}}$.
*   **İdeal Senari:** PPL dəyəri nə qədər kiçik olsa, model o qədər yaxşıdır. Məsələn, PPL=10 o deməkdir ki, model hər növbəti token üçün 10 bərabər ehtimal olunan seçim arasında qərar verir.

## 26.3. Praktika: Monitorinqin Tətbiqi

Biz monitorinq üçün **TensorBoard** və ya **Weights & Biases (W&B)** kimi alətlərdən istifadə edə bilərik. Sadəlik üçün, biz nəticələri hər addımda terminala çap edəcəyik və modelin keyfiyyətini əl ilə izləyəcəyik.

**`train_accelerate.py` skriptində dəyişikliklər:**

```python
# ... (Əvvəlki kodlar) ...

# Təlim dövrü
for step, batch in enumerate(train_dataloader):
    # ... (Forward pass və loss hesablanması) ...
    
    # Loss-u geri yaymaq (Backpropagation)
    accelerator.backward(loss)
    
    # Qradiyentləri yeniləmək
    optimizer.step()
    optimizer.zero_grad()
    
    # ------------------------------------------------
    # 1. Monitorinq: Hər 100 addımda nəticəni çap etmək
    if step % 100 == 0:
        # Loss-u CPU-ya köçürüb rəqəmə çevirmək
        current_loss = loss.item()
        # Perplexity hesablamaq
        perplexity = torch.exp(torch.tensor(current_loss))
        
        # Terminala çap etmək
        print(f"Epoch {epoch} | Step {step}/{len(train_dataloader)} | Loss: {current_loss:.4f} | PPL: {perplexity:.2f}")
        
        # 2. Validasiya Loss-unun Hesablanması (Hər 1000 addımda)
        if step % 1000 == 0 and step > 0:
            val_loss = estimate_loss(model, val_dataloader, accelerator)
            val_ppl = torch.exp(torch.tensor(val_loss))
            print(f"--- Validasiya Nəticəsi ---")
            print(f"Validasiya Loss: {val_loss:.4f} | Validasiya PPL: {val_ppl:.2f}")
            print(f"---------------------------")
            
# ... (estimate_loss funksiyası) ...
@torch.no_grad() # Bu funksiyada qradiyentləri hesablamağa ehtiyac yoxdur
def estimate_loss(model, dataloader, accelerator):
    model.eval() # Modeli qiymətləndirmə rejiminə keçirmək
    losses = []
    for batch in dataloader:
        # ... (Forward pass və loss hesablanması) ...
        # Loss-u CPU-ya köçürüb siyahıya əlavə etmək
        losses.append(accelerator.gather(loss).mean().item())
    model.train() # Modeli təlim rejiminə qaytarmaq
    return np.mean(losses)
```

## 26.4. Overfitting (Həddindən Artıq Öyrənmə)

Monitorinq zamanı ən çox diqqət etməli olduğunuz məqam **Overfitting**-dir:

> **Overfitting:** Təlim Loss-u azalır, lakin Validasiya Loss-u artmağa başlayır.

Bu o deməkdir ki, model təlim məlumatlarını əzbərləyir, lakin yeni məlumatlar üzərində ümumiləşdirmə qabiliyyətini itirir. Overfitting baş verdikdə, təlimi dayandırmaq və ya **Dropout** kimi tənzimləmə (Regularization) texnikalarını artırmaq lazımdır.

**Gündəlik Tapşırıq:** `train_accelerate.py` skriptinə `estimate_loss` funksiyasını və monitorinq kodlarını əlavə edin. Təlimi başlatdıqdan sonra, terminalda Loss və PPL dəyərlərinin necə dəyişdiyini izləyin.
