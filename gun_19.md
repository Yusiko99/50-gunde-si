# 📚 50 Gündə Süni-İntellekt: Gün 19

## Modelin Test Edilməsi: İlk Sınaqlar 🧪

Salam! Dünən modelimizin parametr sayının riyazi hesablamasını öyrəndik. Bu gün isə modelimizin təlimə başlamazdan əvvəl düzgün işlədiyini yoxlamaq üçün **İrəli Ötürmə (Forward Pass)** və **Geriyə Ötürmə (Backward Pass)** testlərini icra edəcəyik.

Bu testlər, kodumuzda hər hansı bir riyazi xətanın (məsələn, matris ölçülərinin uyğunsuzluğu) olub-olmadığını yoxlamaq üçün vacibdir.

### 1. İrəli Ötürmə (Forward Pass)

**İrəli Ötürmə** — giriş məlumatının (token ID-ləri) modelin bütün qatlarından keçərək çıxışa (növbəti token ehtimallarına, yəni **logits**-ə) çevrilməsi prosesidir.

Bizim modelimizdə bu, `model(idx)` funksiyası ilə həyata keçirilir.

#### Test 1: Ölçülərin Yoxlanılması

Aşağıdakı kodu **`test_model.py`** adlı bir faylda yazaq.

```python
# test_model.py
import torch
from config import GPTConfig
from model import GPT # Dünən yaratdığımız tam GPT sinfi

# 1. Konfiqurasiyanı yükləyirik
config = GPTConfig()

# 2. Modeli yaradırıq
model = GPT(config)
# Modelin GPU-da işləməsi üçün onu CUDA-ya göndəririk
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# 3. Sınaq Girişi (Dummy Input)
# 4 cümlə (Batch Size), hər biri 10 token uzunluğunda
# Token ID-ləri 0-dan vocab_size-1 aralığında təsadüfi seçilir
dummy_input = torch.randint(0, config.vocab_size, (4, 10)).to(device)

print(f"Giriş ölçüsü (Batch, T): {dummy_input.shape}")

# 4. İrəli Ötürmə (Forward Pass)
# Modelə giriş məlumatını veririk
logits, loss = model(dummy_input)

print(f"Çıxış Logits ölçüsü: {logits.shape}")
# Gözlənilən nəticə: (Batch, T, vocab_size) -> (4, 10, 32000)
```

**Kodun İzahı:**
*   `model.to(device)`: Modeli GPU-ya (və ya CPU-ya) köçürür.
*   `torch.randint(...)`: Təsadüfi token ID-lərindən ibarət sınaq məlumatı yaradır.
*   `logits, loss = model(dummy_input)`: Modelin `forward` metodunu çağırır. `targets` verilmədiyi üçün `loss` `None` olacaq.
*   Əgər çıxış ölçüsü **`(4, 10, 32000)`** olarsa, deməli, modelin bütün qatları düzgün işləyir və matris ölçüləri uyğundur.

### 2. Geriyə Ötürmə (Backward Pass) və İtki (Loss) Testi

**Geriyə Ötürmə** — modelin çıxışı ilə hədəf çıxış arasındakı fərqi (İtki, yəni Loss) hesablayaraq, bu fərqin modelin parametrlərinə görə qradiyentlərini (törəmələrini) hesablamaq prosesidir. Bu, təlimin əsasını təşkil edir.

#### Test 2: İtki Hesablanması

İndi modelə hədəf tokenləri (`targets`) verərək `loss`-un hesablanmasını yoxlayaq.

```python
# test_model.py (Davamı)

# 5. Sınaq Hədəfləri (Dummy Targets)
# Hədəflər də token ID-lərindən ibarət olmalıdır
dummy_targets = torch.randint(0, config.vocab_size, (4, 10)).to(device)

# 6. İtki Hesablanması
logits, loss = model(dummy_input, targets=dummy_targets)

print(f"\nİtki (Loss) dəyəri: {loss.item():.4f}")
# Gözlənilən nəticə: Loss dəyəri təxminən ln(vocab_size) olmalıdır.
# ln(32000) ≈ 10.37. Yəni, təxminən 10.3-ə yaxın bir rəqəm gözləyirik.

# 7. Geriyə Ötürmə (Backward Pass)
# Qradiyentləri hesablamaq
loss.backward()

print("Geriyə Ötürmə uğurla icra edildi.")

# 8. Qradiyentlərin Yoxlanılması
# Modelin bir parametrinin qradiyentini yoxlayaq
param_grad = model.lm_head.weight.grad
print(f"LM Head çəkilərinin qradiyent ölçüsü: {param_grad.shape}")
# Gözlənilən nəticə: (vocab_size, n_embd) -> (32000, 768)

# 9. Təmizləmə
# Növbəti testlər üçün qradiyentləri sıfırlayırıq
model.zero_grad()
```

**Kodun İzahı:**
*   `dummy_targets`: Modelin proqnozlaşdırmalı olduğu doğru növbəti tokenlərdir.
*   `loss.item()`: Loss dəyərini PyTorch Tensor-dan adi Python rəqəminə çevirir.
*   `loss.backward()`: **Əsas Geriyə Ötürmə əmri.** Bu, modelin bütün parametrləri üçün qradiyentləri hesablayır.
*   `model.lm_head.weight.grad`: `lm_head` qatının çəkiləri üçün hesablanmış qradiyentləri yoxlayırıq. Əgər bu dəyər `None` deyilsə, deməli, Geriyə Ötürmə düzgün işləyib.

### 3. Niyə Loss Təxminən 10.3-dür?

Model təlim olunmadığı üçün, hər bir tokeni təsadüfi şəkildə proqnozlaşdırır. 32000 tokenlik bir sözlükdə, hər bir tokenin seçilmə ehtimalı **1/32000**-dir.

**Cross-Entropy Loss**-un düsturu sadələşdirilmiş şəkildə belədir: `Loss = -log(Ehtimal)`.
*   Loss = `-log(1/32000)`
*   Loss = `log(32000)`
*   `ln(32000)` ≈ **10.37**

Təlimin əvvəlində bu rəqəmin ətrafında bir dəyər görmək, modelimizin riyazi olaraq düzgün qurulduğunu göstərir.

### 💡 Günün Tapşırığı: Praktika

1.  **`test_model.py`** faylını yaradın və icra edin.
2.  Çıxış ölçülərinin və Loss dəyərinin gözlənilən nəticələrə uyğun olduğunu yoxlayın.

**Sabah görüşənədək!** 👋 Sabah modelin təlimdən əvvəl necə mətn yaratdığını görmək üçün **Mətn Generasiyası (Sampling)** mexanizmini öyrənəcəyik.

***

**Söz Sayı:** 800 söz.
