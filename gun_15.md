# Gün 15: Çoxbaşlı Diqqət (Multi-Head Attention) 🤯

## 15.1. Tək Başlı Diqqətin Məhdudiyyəti

Dünən biz **Tək Başlı Diqqət (Single Attention Head)** mexanizmini öyrəndik. Bu mexanizm modelə bir sözün digər sözlərlə olan **bir növ** əlaqəsini tapmağa kömək edir. Lakin dil çox mürəkkəbdir və bir sözün eyni anda bir neçə fərqli əlaqəsi ola bilər:

*   **Sintaktik Əlaqə:** Cümlənin qrammatik quruluşu.
*   **Semantik Əlaqə:** Sözlərin mənası.
*   **Referensial Əlaqə:** Əvəzliklərin aid olduğu isimlər.

Tək bir diqqət başı bütün bu əlaqələri eyni anda öyrənməkdə çətinlik çəkir.

## 15.2. Çoxbaşlı Diqqət (Multi-Head Attention - MHA)

**Çoxbaşlı Diqqət** bu problemi həll edir. O, sadəcə olaraq, diqqət mexanizmini **paralel şəkildə bir neçə dəfə** (bizim modelimizdə 12 dəfə) icra edir.

*   Hər bir **"baş"** (Head) fərqli bir əlaqə növünü öyrənməyə fokuslanır.
*   Məsələn, bir baş sintaktik əlaqəyə, digəri isə semantik əlaqəyə diqqət yetirə bilər.

**MHA-nın İş Prinsipi:**

1.  **Paralel Hesablama:** Giriş məlumatı eyni anda **N sayda** (bizim halda 12) müstəqil diqqət başına göndərilir.
2.  **Nəticələrin Birləşdirilməsi:** Hər bir baş öz nəticəsini (V matrisinin çəkili cəmi) çıxarır.
3.  **Xətti Lay:** Bütün nəticələr birləşdirilir (Concatenate) və yekun bir **Xətti Lay (Linear Layer)**-dən keçirilərək modelin əsas ölçüsünə (768) qaytarılır.

Bu, modelə eyni anda mətnin müxtəlif aspektlərinə **"diqqət yetirməyə"** imkan verir.

## 15.3. Praktika: Multi-Head Attention-ın Qurulması

Dünənki `Head` sinfini istifadə edərək `MultiHeadAttention` sinfini quraq.

**`multi_head_attention.py`**

```python
import torch
import torch.nn as nn
from torch.nn import functional as F
# Dünənki Head sinfini bura kopyalayın və ya import edin

# Modelin əsas hiperparametrləri (Gün 13-dən)
n_embd = 768  # Embedding ölçüsü
n_head = 12   # Başların sayı
block_size = 256 # Kontekst uzunluğu

# Head sinfi (Gün 14-dən)
class Head(nn.Module):
    # ... (Head sinfinin kodu olduğu kimi qalır) ...
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   
        q = self.query(x) 
        
        head_size = k.shape[-1]
        wei = q @ k.transpose(-2, -1) * head_size**-0.5 
        
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        
        wei = F.softmax(wei, dim=-1) 
        
        v = self.value(x) 
        out = wei @ v     
        
        return out


class MultiHeadAttention(nn.Module):
    """Çoxbaşlı Diqqət Mexanizmi"""
    
    def __init__(self, num_heads, head_size):
        super().__init__()
        # N sayda (12) Head sinfini paralel şəkildə yaradırıq
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        
        # Bütün nəticələri birləşdirdikdən sonra tətbiq olunacaq yekun xətti lay
        self.proj = nn.Linear(n_embd, n_embd)
        
        # RTX 2050 üçün kritik: Dropout
        # Təlim zamanı neyronların bir hissəsini təsadüfi olaraq söndürür.
        # Bu, modelin həddindən artıq öyrənməsinin (Overfitting) qarşısını alır.
        self.dropout = nn.Dropout(0.1) 

    def forward(self, x):
        # 1. Bütün başları paralel icra etmək
        # Nəticə: [ (B, T, head_size), (B, T, head_size), ... ]
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        
        # 2. Birləşdirilmiş nəticəni yekun xətti laydan keçirmək
        out = self.dropout(self.proj(out))
        
        return out

# Nümunə: Multi-Head Attention yaratmaq
mha = MultiHeadAttention(num_heads=n_head, head_size=n_embd // n_head)
print(mha)
```

## 15.4. Kodun İzahı

| Sətr | Kod | İzahı |
| :--- | :--- | :--- |
| **49** | `self.heads = nn.ModuleList([...])` | **`nn.ModuleList`** PyTorch-da bir neçə eyni sinfi (bizim halda 12 ədəd `Head` sinfini) bir siyahıda saxlamağa imkan verir. |
| **52** | `self.proj = nn.Linear(n_embd, n_embd)` | Bütün 12 başın nəticəsi birləşdirildikdən sonra, bu lay nəticəni yenidən modelin əsas ölçüsünə (768) qaytarır. |
| **56** | `self.dropout = nn.Dropout(0.1)` | **Dropout** təlimi sabitləşdirmək üçün vacibdir. 0.1 o deməkdir ki, hər addımda neyronların 10%-i təsadüfi olaraq söndürüləcək. |
| **60** | `out = torch.cat([h(x) for h in self.heads], dim=-1)` | Bütün 12 başın çıxışını **sonuncu ölçü (dim=-1)** üzrə birləşdirir (Concatenate). Nəticənin ölçüsü: (Batch, Time, 12 * 64) = (B, T, 768). |
| **62** | `out = self.dropout(self.proj(out))` | Birləşdirilmiş nəticəni `proj` layından keçirir və Dropout tətbiq edir. |

**Gündəlik Tapşırıq:** `multi_head_attention.py` skriptini yaradın. `torch.cat` əməliyyatının nəticənin ölçüsünü necə dəyişdiyini anlamağa çalışın. Bu, Transformer Blokunun əsasını təşkil edir.
