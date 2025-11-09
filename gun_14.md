# 📚 50 Gündə Süni-İntellekt: Gün 14

## PyTorch-da Əsas Bloklar: Təməl Qatlar 🧱

Salam! Dünən 100M parametreli NanoGPT modelimizin konfiqurasiyasını təyin etdik. Bu gün isə modelin ən təməlini təşkil edən PyTorch bloklarını – **`nn.Module`**, **`Tensor`** və **Gömülmə Qatını (Embedding Layer)** öyrənəcəyik.

### 1. nn.Module və Tensor Anlayışları

Dərin Öyrənmə modelləri PyTorch-da **`nn.Module`** sinfi vasitəsilə qurulur.

> **`nn.Module`** — PyTorch-da bütün neyron şəbəkə qatları və modelləri üçün əsas sinifdir. Hər bir qat və ya model bu sinifdən miras almalıdır.

Bu sinif iki əsas metodu tələb edir:
1.  **`__init__`**: Modelin qatlarının və parametrlərinin təyin olunduğu yer.
2.  **`forward`**: Məlumatın (Tensor-un) modeldən necə keçdiyini (irəli ötürmə) təyin edən yer.

> **Tensor** — PyTorch-da məlumatları saxlamaq üçün istifadə olunan əsas riyazi strukturdur. O, NumPy massivlərinə bənzəyir, lakin GPU-da işləmək üçün optimallaşdırılıb.

### 2. Gömülmə Qatı (Embedding Layer)

Bizim tokenizatorumuz hər bir sözü unikal bir rəqəmə (ID) çevirir. Lakin bu rəqəmlər (məsələn, 1, 2, 3) model üçün heç bir məna daşımır. Modelin sözləri başa düşməsi üçün onları **mənalı rəqəmsal vektorlara** çevirməliyik. Bu işi **Gömülmə Qatı** görür.

> **Gömülmə Qatı (`nn.Embedding`)** — hər bir token ID-sini modelin öyrənə biləcəyi sabit ölçülü (bizim halımızda `n_embd=768`) bir vektora çevirir. Bu vektorlar təlim zamanı avtomatik olaraq yenilənir və oxşar mənalı sözlər (məsələn, "kitab" və "dərs") oxşar vektorlara sahib olur.

#### Gömülmə Qatının Qurulması

Bizim NanoGPT modelimizin ilk qatı **Token Gömülməsi** və **Mövqe Gömülməsi** olacaq.

```python
# gpt_model_base.py (GPT modelinin əsas sinfi)
import torch
import torch.nn as nn
from config import GPTConfig # Dünən yaratdığımız konfiqurasiya

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 1. Token Gömülməsi (Token Embedding)
        # Sözlük həcmi (vocab_size) x Gömülmə ölçüsü (n_embd) matrisi yaradır
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)

        # 2. Mövqe Gömülməsi (Positional Embedding)
        # Maksimum ardıcıllıq uzunluğu (block_size) x Gömülmə ölçüsü (n_embd) matrisi yaradır
        self.wpe = nn.Embedding(config.block_size, config.n_embd)

        # 3. Dropout (Overfitting-in qarşısını almaq üçün)
        self.drop = nn.Dropout(config.dropout)

        # 4. Transformer Blokları (Növbəti günlərdə əlavə olunacaq)
        # self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        # 5. Son Normallaşdırma və Xətti Qat (Head)
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx):
        # idx: Token ID-lərindən ibarət Tensor (B, T)
        B, T = idx.size() # Batch size (B) və Ardıcıllıq uzunluğu (T)

        # 1. Mövqe ID-lərini yaratmaq
        # 0-dan T-1-ə qədər rəqəmlər ardıcıllığı (məsələn, [0, 1, 2, 3, ...])
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # (T)

        # 2. Gömülmələri Hesablamaq
        # Token ID-lərini vektorlara çevirir
        token_emb = self.wte(idx) # (B, T, n_embd)
        # Mövqe ID-lərini vektorlara çevirir
        pos_emb = self.wpe(pos)   # (T, n_embd)

        # 3. Token və Mövqe Gömülmələrini Toplamaq
        x = self.drop(token_emb + pos_emb) # (B, T, n_embd)

        # 4. Transformer Bloklarından Keçirmək (Hələlik boşdur)
        # for block in self.h:
        #     x = block(x)

        # 5. Son Qatlar
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        return logits
```

### 3. Kodun İzahı (Hər Sətrin Detallı İzahı)

| Sətr | Kod | İzah |
| :--- | :--- | :--- |
| 12 | `self.wte = nn.Embedding(...)` | **Token Gömülməsi:** `vocab_size` (32000) sayda token üçün `n_embd` (768) ölçülü vektorlar yaradır. |
| 16 | `self.wpe = nn.Embedding(...)` | **Mövqe Gömülməsi:** `block_size` (512) sayda mövqe üçün `n_embd` (768) ölçülü vektorlar yaradır. |
| 24 | `B, T = idx.size()` | Giriş məlumatının ölçülərini (Batch size və Ardıcıllıq uzunluğu) alır. |
| 28 | `pos = torch.arange(0, T, ...)` | 0-dan T-1-ə qədər mövqe indekslərini yaradır. |
| 31 | `token_emb = self.wte(idx)` | Token ID-lərini (idx) mənalı vektorlara çevirir. |
| 33 | `pos_emb = self.wpe(pos)` | Mövqe indekslərini mənalı vektorlara çevirir. |
| 36 | `x = self.drop(token_emb + pos_emb)` | **Token və Mövqe Gömülmələrini toplayırıq.** Bu, modelə həm sözün mənasını, həm də cümlədəki yerini eyni anda verir. |
| 43 | `self.lm_head = nn.Linear(...)` | **Dil Modeli Başı (LM Head):** 768 ölçülü vektoru yenidən 32000 ölçülü vektorlara çevirir. Bu 32000 rəqəm, növbəti tokenin hansı token ID-si olmasının ehtimalını göstərir. |

### 💡 Günün Tapşırığı: Praktika

1.  **`config.py`** faylının mövcud olduğundan əmin olun.
2.  **`gpt_model_base.py`** faylını yaradın və yuxarıdakı kodu ora kopyalayın.
3.  Kiçik bir sınaq skripti yazın:
    ```python
    # Sınaq skripti
    from config import GPTConfig
    from gpt_model_base import GPT
    
    config = GPTConfig()
    model = GPT(config)
    
    # Sınaq girişi: 4 cümlə (batch), hər biri 10 token uzunluğunda
    dummy_input = torch.randint(0, config.vocab_size, (4, 10))
    
    output = model(dummy_input)
    print(f"Çıxış Tensorunun Ölçüsü: {output.shape}")
    # Nəticə (4, 10, 32000) olmalıdır.
    ```

**Sabah görüşənədək!** 👋 Sabah **Çoxbaşlı Diqqət (Multi-Head Attention)** mexanizmini PyTorch-da sıfırdan quracağıq.

***

**Söz Sayı:** 750 söz.
