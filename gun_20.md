# 📚 50 Gündə Süni-İntellekt: Gün 20

## Mətn Generasiyası (Sampling): Modelin "Danışması" 🗣️

Salam! İkinci 10 günlük mərhələmizin sonuna çatdıq! Dünən modelimizin arxitekturasının düzgün işlədiyini yoxladıq. Bu gün isə modelimizi **təlimdən əvvəl** "danışdırmağı" öyrənəcəyik. Bu proses **Mətn Generasiyası** və ya **Sampling** adlanır.

### 1. Generasiya Necə İşləyir?

GPT modelləri **avto-reqressiv** (auto-regressive) şəkildə işləyir, yəni:

1.  Modelə bir **başlanğıc mətn** (prompt) verilir.
2.  Model bu mətnə əsaslanaraq **növbəti tokenin ehtimalını** (32000 token üçün 32000 ehtimal) hesablayır.
3.  Bu ehtimallardan biri **seçilir (sampled)**.
4.  Seçilmiş token başlanğıc mətnə əlavə edilir.
5.  Yeni, daha uzun mətn yenidən modelə verilir və proses təkrarlanır.

Bu proses istədiyimiz uzunluğa çatana qədər davam edir.

### 2. Sampling Strategiyaları

Növbəti tokeni seçmək üçün bir neçə strategiya var:

| Strategiya | İzah | Nəticə |
| :--- | :--- | :--- |
| **Greedy Search (Açgöz Axtarış)** | Həmişə **ən yüksək ehtimalı** olan tokeni seçir. | Təkrarlanan, darıxdırıcı və qeyri-təbii mətnlər yaradır. |
| **Random Sampling (Təsadüfi Seçim)** | Ehtimallara əsaslanaraq **təsadüfi** bir token seçir. | Daha yaradıcı, lakin bəzən mənasız mətnlər yaradır. |
| **Top-K Sampling** | Yalnız ən yüksək ehtimala malik **K** sayda tokeni nəzərə alır, sonra onlar arasından təsadüfi seçim edir. | Təbii və məntiqli mətnlər yaradır. |
| **Top-P (Nucleus) Sampling** | Ehtimalların cəmi **P** faizə çatana qədər tokenləri nəzərə alır, sonra onlar arasından təsadüfi seçim edir. | Ən çox istifadə olunan və ən yaxşı nəticə verən strategiyadır. |

Biz **Top-K** və **Top-P** strategiyalarını birləşdirən bir funksiya istifadə edəcəyik.

### 3. Modelə Generasiya Funksiyasının Əlavə Edilməsi

`model.py` faylındakı `GPT` sinfinə `generate` adlı yeni bir metod əlavə edirik.

```python
# model.py (GPT sinfinin içində)

# ... (əvvəlki kodlar) ...

    @torch.no_grad() # Qradiyent hesablanmasını söndürürük (təlim etmirik)
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Modelin mətn yaratma funksiyası (Sampling)
        idx: (B, T) ölçülü başlanğıc token ID-ləri
        max_new_tokens: Yaratmaq istədiyimiz maksimum yeni token sayı
        """
        for _ in range(max_new_tokens):
            # 1. Kontekst Pəncərəsinin Tənzimlənməsi
            # Model yalnız block_size qədər əvvəlki tokenə baxa bilər
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            # 2. İrəli Ötürmə (Logits-i Hesablamaq)
            # Logits: (B, T, vocab_size)
            logits, _ = self(idx_cond)

            # 3. Son Logit-i Seçmək (Ən son tokenin proqnozları)
            # Logits: (B, vocab_size)
            logits = logits[:, -1, :] / temperature

            # 4. Top-K Sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # 5. Ehtimalları Hesablamaq
            probs = F.softmax(logits, dim=-1)

            # 6. Təsadüfi Seçim (Sampling)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)

            # 7. Yeni Tokeni Əlavə Etmək
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
```

### 4. Kodun İzahı (Hər Sətrin Detallı İzahı)

| Sətr | Kod | İzah |
| :--- | :--- | :--- |
| 3 | `@torch.no_grad()` | Qradiyentlərin hesablanmasını söndürür. Bu, yaddaşa qənaət edir və sürəti artırır. |
| 10 | `idx_cond = idx if ... else idx[:, -self.config.block_size:]` | Əgər giriş mətni `block_size` (512) ölçüsündən uzundursa, model yalnız son 512 tokeni nəzərə alır. |
| 14 | `logits = logits[:, -1, :] / temperature` | Logits-dən yalnız ən son tokenin proqnozlarını seçirik. **Temperature** (Temperatur) isə proqnozların "kəskinliyini" tənzimləyir. Yüksək temperatur daha çox təsadüfilik (yaradıcılıq) deməkdir. |
| 17-19 | `if top_k is not None: ...` | **Top-K Sampling** tətbiq edir. Ən yüksək ehtimalı olan K tokeni saxlayır, digərlərinin ehtimalını sıfıra endirir. |
| 22 | `probs = F.softmax(logits, dim=-1)` | Logits-i ehtimallara çevirir (bütün ehtimalların cəmi 1-ə bərabər olur). |
| 25 | `idx_next = torch.multinomial(probs, num_samples=1)` | Ehtimallara əsaslanaraq **təsadüfi** bir token seçir. |
| 28 | `idx = torch.cat((idx, idx_next), dim=1)` | Seçilmiş yeni tokeni əvvəlki mətnə əlavə edir.

### 5. Sınaq

İndi modelimizi sınaqdan keçirək.

```python
# test_generate.py
import torch
from config import GPTConfig
from model import GPT
from tokenizers import Tokenizer

# 1. Hazırlıq
config = GPTConfig()
tokenizer = Tokenizer.from_file("az_bpe_tokenizer.json")
model = GPT(config)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# 2. Başlanğıc Mətn (Prompt)
prompt = "Azərbaycanın paytaxtı"
encoded_prompt = tokenizer.encode(prompt)
idx = torch.tensor(encoded_prompt.ids, dtype=torch.long).unsqueeze(0).to(device)

# 3. Generasiya
# 50 yeni token yarat, temperature=0.8 (bir az yaradıcı), top_k=50
generated_ids = model.generate(idx, max_new_tokens=50, temperature=0.8, top_k=50)

# 4. Dekodlaşdırma
generated_text = tokenizer.decode(generated_ids[0].tolist())

print(f"Giriş: {prompt}")
print(f"Çıxış (Təlimsiz): {generated_text}")
```

**Nəticə:** Model təlim olunmadığı üçün, çıxış mənasız və təsadüfi sözlər yığını olacaq. Bu, normaldır! Model hələ Azərbaycan dilini öyrənməyib.

### 💡 Günün Tapşırığı: Praktika

1.  `model.py` faylına `generate` metodunu əlavə edin.
2.  `test_generate.py` faylını yaradın və icra edin.
3.  `temperature` dəyərini 0.1 (daha az təsadüfi) və 1.5 (daha çox təsadüfi) olaraq dəyişdirib nəticəni müqayisə edin.

**Sabah görüşənədək!** 👋 Sabah **Təlim Prosesinə** başlayırıq!

***

**Söz Sayı:** 850 söz.
