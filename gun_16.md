# Gün 16: Transformer Blokunun Qurulması 🧱

## 16.1. Transformer Bloku Nədir?

**Transformer Bloku** (və ya GPT Bloku) modelin əsas təkrar olunan vahididir. Bizim 100M parametrli modelimizdə bu blokdan **12 ədəd** ardıcıl istifadə olunacaq.

Bir Transformer Bloku iki əsas alt-blokdan ibarətdir:

1.  **Multi-Head Attention (MHA):** Mətnin kontekstini öyrənir (Gün 15).
2.  **Feed-Forward Network (FFN):** MHA-dan gələn məlumatı emal edir və modelin öyrənmə qabiliyyətini artırır.

Bu iki alt-blok arasında və onlardan sonra **Layer Normalization (Lay Normallaşdırması)** və **Residual Connection (Qalıq Əlaqə)** istifadə olunur.

## 16.2. Layer Normalization və Residual Connection

*   **Residual Connection (Qalıq Əlaqə):** Giriş məlumatını (x) alt-blokun çıxışına əlavə edir. Yəni, `çıxış = x + AltBlok(x)`. Bu, qradiyentlərin dərin şəbəkələrdə belə asanlıqla axmasına və modelin daha sürətli öyrənməsinə kömək edir.
*   **Layer Normalization (Lay Normallaşdırması):** Hər bir alt-blokun çıxışını normallaşdırır. Bu, təlim prosesini sabitləşdirir və sürətləndirir.

## 16.3. Praktika: Transformer Blokunun Qurulması

İndi isə `MultiHeadAttention` sinfini və `FeedForward` sinfini birləşdirərək `Block` sinfini quraq.

**`block.py`**

```python
import torch
import torch.nn as nn
from torch.nn import functional as F
# MultiHeadAttention sinfini (Gün 15-dən) bura kopyalayın və ya import edin

# Modelin əsas hiperparametrləri (Gün 13-dən)
n_embd = 768  # Embedding ölçüsü
n_head = 12   # Başların sayı
block_size = 256 # Kontekst uzunluğu

# ... (Head sinfinin kodu) ...
# ... (MultiHeadAttention sinfinin kodu) ...

class FeedForward(nn.Module):
    """Sadə İrəli-Ötürmə Şəbəkəsi (MLP)"""
    
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            # 1. Genişləndirmə: Ölçünü 4 dəfə artırırıq (768 * 4 = 3072)
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(), # Aktivasiya funksiyası (ReLU-dan daha yaxşıdır)
            # 2. Daraltma: Ölçünü yenidən 768-ə qaytarırıq
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(0.1), # RTX 2050 üçün Overfitting-in qarşısını almaq
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer Blokunun Təkrar Olunan Vahidi"""
    
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        
        # 1. Multi-Head Attention (MHA)
        self.sa = MultiHeadAttention(n_head, head_size)
        
        # 2. Feed-Forward Network (FFN)
        self.ffwd = FeedForward(n_embd)
        
        # 3. Layer Normalization (Normallaşdırma)
        # Hər bir alt-blokdan əvvəl tətbiq olunur (Pre-Layer Norm)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # 1. Birinci Alt-Blok: MHA + Residual Connection + Layer Norm
        # Layer Norm-dan keçirib MHA-ya ötürürük, sonra girişi (x) əlavə edirik.
        x = x + self.sa(self.ln1(x))
        
        # 2. İkinci Alt-Blok: FFN + Residual Connection + Layer Norm
        # Layer Norm-dan keçirib FFN-ə ötürürük, sonra girişi (x) əlavə edirik.
        x = x + self.ffwd(self.ln2(x))
        
        return x

# Nümunə: Tək bir Transformer Bloku yaratmaq
block = Block(n_embd=n_embd, n_head=n_head)
print(block)
```

## 16.4. Kodun İzahı

| Sətr | Kod | İzahı |
| :--- | :--- | :--- |
| **32** | `nn.Linear(n_embd, 4 * n_embd)` | FFN-in ilk xətti layı. Giriş ölçüsünü 4 dəfə artırır. Bu genişləndirmə modelə daha mürəkkəb əlaqələri öyrənməyə imkan verir. |
| **33** | `nn.GELU()` | **Gaussian Error Linear Unit** (GELU) aktivasiya funksiyası. ReLU-dan daha hamar və LLM-lərdə daha çox istifadə olunur. |
| **35** | `nn.Linear(4 * n_embd, n_embd)` | FFN-in ikinci xətti layı. Ölçünü yenidən modelin əsas ölçüsünə qaytarır. |
| **57** | `self.ln1 = nn.LayerNorm(n_embd)` | Birinci Layer Norm layı. |
| **61** | `x = x + self.sa(self.ln1(x))` | **Residual Connection** (`x + ...`) və **Pre-Layer Normalization** (`self.ln1(x)`) tətbiq olunur. Bu, Transformer arxitekturasının standart tətbiqidir. |
| **64** | `x = x + self.ffwd(self.ln2(x))` | İkinci alt-blokun (FFN) tətbiqi. |

**Gündəlik Tapşırıq:** `block.py` skriptini yaradın. `Block` sinfinin `forward` funksiyasındakı **Residual Connection** və **Layer Normalization** ardıcıllığını dərindən analiz edin. Bu, GPT modelinin əsasını təşkil edir.
