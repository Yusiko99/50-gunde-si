# Gün 26: Təlimin Monitorinqi və Vizualizasiyası 📊

## 26.1. Monitorinqin Əhəmiyyəti

LLM təlimi, xüsusilə məhdud resurslarda, uzun və resurs-tələbkar bir prosesdir. Təlimin gedişatını **monitorinq etmək** modelin düzgün öyrəndiyini, Overfitting (həddindən artıq əzbərləmə) riskinin olub-olmadığını və optimallaşdırmanın effektivliyini yoxlamaq üçün zəruridir.

**Əsas Monitorinq Metrikaları:**

1.  **Təlim Loss-u (Training Loss):** Modelin təlim məlumatları üzərindəki səhvi.
2.  **Validasiya Loss-u (Validation Loss):** Modelin görmədiyi məlumatlar üzərindəki səhvi.
3.  **Öyrənmə Sürəti (Learning Rate):** Optimizerin hər addımda çəkiləri nə qədər dəyişdirdiyi.

## 26.2. Perplexity (PPL) Metrikası

Loss dəyəri riyazi bir ölçü olsa da, **Perplexity (PPL)** modelin dil üzərindəki qabiliyyətini daha intuitiv şəkildə ifadə edir.

*   **Nədir?** PPL, Loss-un eksponensial funksiyasıdır: $PPL = e^{Loss}$.
*   **Məntiq:** PPL modelin növbəti tokeni proqnozlaşdırmaqda nə qədər "çaşqın" olduğunu göstərir. PPL dəyəri nə qədər aşağı olarsa, modelin proqnozlaşdırması bir o qədər dəqiqdir. Məsələn, PPL=10 o deməkdir ki, model hər növbəti token üçün orta hesabla 10 bərabər ehtimallı seçim arasında qalır.

## 26.3. Vizualizasiya üçün `TensorBoard`

Təlim metrikalarını vizual şəkildə izləmək üçün **TensorBoard** ən geniş yayılmış alətdir.

**TensorBoard-un İnteqrasiyası:**

1.  **Quraşdırma:** `pip install tensorboard`
2.  **`SummaryWriter`:** PyTorch-da `torch.utils.tensorboard.SummaryWriter` istifadə edərək metrikaları log fayllarına yazmaq.

**`train_accelerate.py` Skriptinə Əlavələr:**

```python
# ... (Əvvəlki kod) ...
from torch.utils.tensorboard import SummaryWriter

# 1. Konfiqurasiya
# ...
LOG_DIR = "runs/az_llm_experiment_1"
writer = SummaryWriter(LOG_DIR)
global_step = 0

# ... (Təlim dövrü) ...

for epoch in range(NUM_EPOCHS):
    model.train()
    for step, batch in enumerate(train_dataloader):
        # ... (Təlim addımları) ...
        
        # 7. Metrikaların Loglanması
        if step % 10 == 0:
            # Təlim Loss-unu loglamaq
            writer.add_scalar('Loss/Train', loss.item(), global_step)
            
            # Öyrənmə Sürətini loglamaq
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('Learning_Rate', current_lr, global_step)
            
        global_step += 1
        
    # Hər epoxanın sonunda Validasiya Loss-unu loglamaq
    # ... (Gün 27-də əlavə olunacaq) ...

# Təlim bitdikdən sonra
writer.close()
```

## 26.4. TensorBoard-un İşə Salınması

Təlim skripti işləyərkən, başqa bir terminalda TensorBoard-u işə salmaq lazımdır:

```bash
tensorboard --logdir=runs
```

Bu əmr, yerli kompüterdə bir veb-server işə salacaq (adətən `http://localhost:6006`). Bu ünvana daxil olaraq təlimin gedişatını qrafiklər şəklində izləmək mümkündür.

**Məntiq:** Vizualizasiya, təlimin gedişatını bir baxışda anlamağa və Overfitting kimi problemləri erkən aşkar etməyə imkan verir.
