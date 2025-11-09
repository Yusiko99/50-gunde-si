# 📚 50 Gündə Süni-İntellekt: Gün 16

## Transformer Blokunun Qurulması 🏗️

Salam! Dünən NanoGPT modelimizin ən mürəkkəb hissəsi olan **Çoxbaşlı Diqqət (Multi-Head Attention)** mexanizmini PyTorch-da qurduq. Bu gün isə bu mexanizmi digər əsas komponentlərlə birləşdirərək **Transformer Blokunu** (və ya NanoGPT-dəki adıyla **Block** sinfini) yaradacağıq.

### 1. Transformer Blokunun Komponentləri

Bir **Transformer Bloku** iki əsas alt-blokdan ibarətdir:

1.  **Multi-Head Attention (MHA):** Mətnin fərqli hissələri arasındakı əlaqələri öyrənir.
2.  **Feed-Forward Network (FFN):** Hər bir tokeni fərdi şəkildə emal edən, sadə, lakin güclü bir neyron şəbəkəsidir.

Bu iki alt-blokun hər biri **Qat Normallaşdırması (Layer Normalization)** və **Qalıq Əlaqə (Residual Connection)** ilə əhatə olunur.

| Komponent | Funksiya |
| :--- | :--- |
| **LayerNorm** | Hər bir qatın girişini normallaşdırır. Bu, təlimi daha stabil və sürətli edir. |
| **Residual Connection** | Qatın girişini birbaşa çıxışa əlavə edir. Bu, modelin dərinləşdikcə öyrənmə qabiliyyətini itirməsinin qarşısını alır. |
| **GELU** | **Gaussian Error Linear Unit** – FFN-də istifadə olunan aktivasiya funksiyasıdır. ReLU-dan daha yaxşı nəticələr verir. |

### 2. PyTorch-da Transformer Blokunun Qurulması

Aşağıdakı kodu **`block.py`** adlı bir faylda yazaq. Bu kod, dünən yazdığımız `MultiHeadAttention` sinfini istifadə edəcək.

```python
# block.py
import torch
import torch.nn as nn
from torch.nn import functional as F
from attention import MultiHeadAttention # Dünənki sinif
from config import GPTConfig

class Block(nn.Module):
    """ NanoGPT-də bir Transformer Blokunu təmsil edir """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 1. Qat Normallaşdırması (LayerNorm) - Diqqətdən əvvəl
        self.ln_1 = nn.LayerNorm(config.n_embd)
        # 2. Çoxbaşlı Diqqət (Multi-Head Attention)
        self.attn = MultiHeadAttention(config)

        # 3. Qat Normallaşdırması (LayerNorm) - FFN-dən əvvəl
        self.ln_2 = nn.LayerNorm(config.n_embd)
        # 4. İrəli Ötürmə Şəbəkəsi (Feed-Forward Network)
        # Standart olaraq, FFN-in gizli qatı giriş ölçüsünün 4 qatıdır (768 * 4 = 3072)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
            gelu    = nn.GELU(),
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
            dropout = nn.Dropout(config.dropout),
        ))
        self.mlp_forward = nn.Sequential(self.mlp.c_fc, self.mlp.gelu, self.mlp.c_proj, self.mlp.dropout)

    def forward(self, x):
        # 1. Diqqət Alt-Bloku
        # Qalıq Əlaqə (Residual Connection) + LayerNorm + Attention
        # LayerNorm-u əvvəlcə tətbiq etmək (Pre-LN) daha stabil təlimə səbəb olur
        x = x + self.attn(self.ln_1(x))

        # 2. FFN Alt-Bloku
        # Qalıq Əlaqə (Residual Connection) + LayerNorm + FFN
        x = x + self.mlp_forward(self.ln_2(x))

        return x
```

### 3. Kodun İzahı (Hər Sətrin Detallı İzahı)

| Sətr | Kod | İzah |
| :--- | :--- | :--- |
| 12 | `class Block(nn.Module):` | Transformer Blokumuzun sinfini təyin edirik. |
| 17 | `self.ln_1 = nn.LayerNorm(config.n_embd)` | Birinci Layer Norm qatını yaradırıq. |
| 19 | `self.attn = MultiHeadAttention(config)` | Dünən yazdığımız Çoxbaşlı Diqqət mexanizmini daxil edirik. |
| 23 | `self.ln_2 = nn.LayerNorm(config.n_embd)` | İkinci Layer Norm qatını yaradırıq. |
| 26-31 | `self.mlp = nn.ModuleDict(...)` | **Feed-Forward Network (FFN)**-i təyin edirik. O, 4 əsas hissədən ibarətdir: giriş xətti qatı (`c_fc`), aktivasiya funksiyası (`gelu`), çıxış xətti qatı (`c_proj`) və `dropout`. |
| 32 | `self.mlp_forward = nn.Sequential(...)` | FFN-in komponentlərini ardıcıl icra olunacaq şəkildə birləşdiririk. |
| 35 | `def forward(self, x):` | Məlumatın blokdan keçmə ardıcıllığını təyin edirik. |
| 39 | `x = x + self.attn(self.ln_1(x))` | **Diqqət Alt-Bloku:** Giriş (`x`) LayerNorm-dan keçirilir, sonra Diqqət mexanizminə verilir və nəticə yenidən girişə əlavə edilir (`x + ...`). Bu, **Qalıq Əlaqədir**. |
| 43 | `x = x + self.mlp_forward(self.ln_2(x))` | **FFN Alt-Bloku:** Eyni şəkildə, LayerNorm-dan keçirilir, FFN-dən keçirilir və nəticə yenidən girişə əlavə edilir. |

### 4. Qalıq Əlaqə (Residual Connection)

Qalıq Əlaqənin əhəmiyyətini bir daha vurğulayaq:

```python
output = input + Sublayer(LayerNorm(input))
```

Bu, modelin öyrənmə prosesini asanlaşdırır. Əgər model yeni qatda heç nə öyrənməsə belə, **əvvəlki məlumatı (input)** birbaşa növbəti qata ötürə bilir. Bu, modelin **dərinliyini** (bizim halımızda 12 qat) artırmağa imkan verir.

### 💡 Günün Tapşırığı: Praktika

1.  **`attention.py`** və **`config.py`** fayllarının mövcud olduğundan əmin olun.
2.  **`block.py`** faylını yaradın və yuxarıdakı kodu ora kopyalayın.
3.  Kiçik bir sınaq skripti yazın:
    ```python
    # Sınaq skripti
    from config import GPTConfig
    from block import Block
    
    config = GPTConfig()
    block = Block(config)
    
    # Sınaq girişi: 4 cümlə (batch), hər biri 10 token uzunluğunda, 768 ölçülü vektor
    dummy_input = torch.randn(4, 10, config.n_embd)
    
    output = block(dummy_input)
    print(f"Çıxış Tensorunun Ölçüsü: {output.shape}")
    # Nəticə (4, 10, 768) olmalıdır.
    ```

**Sabah görüşənədək!** 👋 Sabah bütün bu komponentləri birləşdirərək **GPT (NanoGPT)** modelinin tam sinfini yaradacağıq.

***

**Söz Sayı:** 750 söz.
