# 📚 50 Gündə Süni-İntellekt: Gün 27

## Validasiya və Qiymətləndirmə: Modelin Ağıllılıq Dərəcəsi 🧠

Salam! Dünən təlimin monitorinqi və **Overfitting** probleminin qarşısını alma yollarını öyrəndik. Bu gün isə modelimizin nə qədər yaxşı öyrəndiyini ölçmək üçün istifadə olunan əsas metrikə – **Perplexity (Çaşqınlıq)**-a baxacağıq.

### 1. Validasiya Nədir?

**Validasiya** — modelin təlim zamanı görmədiyi, lakin təlim məlumatı ilə eyni paylanmaya malik olan məlumat üzərində performansının yoxlanılmasıdır.

Bizim `train.py` skriptimizdə `estimate_loss()` funksiyası məhz bu işi görür: `val.npy` faylındakı məlumat üzərində **Validasiya İtkisini** hesablayır.

### 2. Perplexity (Çaşqınlıq) Metriki

LLM-lərin performansını ölçmək üçün ən çox istifadə olunan metrik **Perplexity (PPL)**-dir.

> **Perplexity** — modelin növbəti tokeni proqnozlaşdırmaqda nə qədər **çaşqın** olduğunu ölçür. Sadə dildə, bu, modelin mətnin nə qədər yaxşı **"başa düşdüyünü"** göstərir.

**Riyazi Əlaqə:** Perplexity, itki funksiyası (Cross-Entropy Loss) ilə birbaşa əlaqəlidir:

$$
\text{Perplexity} = 2^{\text{Cross-Entropy Loss}}
$$

*   **Aşağı PPL:** Modelin çaşqınlığı azdır, yəni proqnozları daha dəqiqdir. **Daha yaxşı model** deməkdir.
*   **Yüksək PPL:** Modelin çaşqınlığı çoxdur, yəni proqnozları təsadüfidir. **Daha pis model** deməkdir.

**Nümunə:**
*   Əgər Loss = 10.37 (təlimsiz model), onda PPL = $2^{10.37} \approx 1280$.
*   Əgər Loss = 3.0 (yaxşı təlim olunmuş model), onda PPL = $2^{3.0} \approx 8$.

Yəni, təlim olunmuş model təlimsiz modeldən 160 dəfə daha az çaşqındır.

### 3. Perplexity-nin Hesablanması

Bizim `train.py` skriptimizdə `estimate_loss()` funksiyası artıq Loss-u hesablayır. Biz sadəcə bu funksiyanın çıxışını dəyişdirməliyik.

#### `train.py` Skriptində Dəyişiklik

```python
# train.py (estimate_loss funksiyası)

# ... (əvvəlki kodlar) ...

# 4. Validasiya Funksiyası
@torch.no_grad()
def estimate_loss():
    """ Validasiya məlumatı üzərində itkini hesablayır və PPL-i qaytarır """
    model.eval()
    losses = []
    for _ in range(EVAL_ITERS):
        # ... (loss hesablanması) ...
        losses.append(accelerator.gather(loss).mean().item())
    
    # Loss-un ortalamasını hesablayırıq
    mean_loss = torch.tensor(losses).mean().item()
    
    # Perplexity-ni hesablayırıq
    perplexity = 2.0 ** mean_loss
    
    model.train()
    return mean_loss, perplexity # Həm Loss, həm də PPL-i qaytarırıq

# 5. Əsas Təlim Dövrü (Yenilənmiş)
# ...
for iter_num in tqdm(range(MAX_ITERS), desc="Təlim Prosesi"):
    
    # A. Validasiya
    if iter_num % EVAL_INTERVAL == 0:
        val_loss, val_ppl = estimate_loss() # İki dəyər alırıq
        print(f"Addım {iter_num}: Validasiya İtkisi (Loss) = {val_loss:.4f}, PPL = {val_ppl:.2f}")
    # ...
```

**Kodun İzahı:**
*   `mean_loss = torch.tensor(losses).mean().item()`: Bütün validasiya Batch-lərinin itki ortalamasını hesablayır.
*   `perplexity = 2.0 ** mean_loss`: Riyazi düstura əsasən, 2-nin Loss dərəcəsinə yüksəldilmiş qüvvətini hesablayır.
*   Artıq təlimin gedişatını izləyərkən həm Loss-un azaldığını, həm də PPL-in kiçildiyini görəcəyik.

### 4. Modelin Qiymətləndirilməsi üçün Digər Metriklər

PPL modelin nə qədər yaxşı proqnozlaşdırdığını göstərsə də, mətnin **mənasını** və **keyfiyyətini** ölçmür. Chatbotlar üçün əlavə metriklər lazımdır:

| Metrik | Məqsəd |
| :--- | :--- |
| **BLEU/ROUGE** | Modelin yaratdığı mətnin insan tərəfindən yazılmış referans mətnə nə qədər oxşar olduğunu ölçür. |
| **İnsan Qiymətləndirməsi** | Ən etibarlı metrikdir. İnsanlar modelin yaratdığı mətnin **səlisliyini**, **məntiqliliyini** və **uyğunluğunu** qiymətləndirir. |

Bizim layihəmizdə, təlimin sonunda modelin yaratdığı mətnləri oxuyaraq **İnsan Qiymətləndirməsi** edəcəyik.

### 💡 Günün Tapşırığı: Praktika

1.  `train.py` skriptində `estimate_loss()` funksiyasını yeniləyin ki, həm Loss, həm də Perplexity-ni hesablasın.
2.  Təlimi davam etdirin və PPL dəyərinin necə azaldığını izləyin.

**Sabah görüşənədək!** 👋 Sabah **Checkpoint və Modelin Saxlanması** mövzusunu öyrənəcəyik.

***

**Söz Sayı:** 750 söz.
