# Gün 14: Diqqət Mexanizmi (Attention) 💡

## 14.1. Diqqət Nədir?

**Diqqət Mexanizmi (Attention Mechanism)** Transformer arxitekturasının ən vacib hissəsidir. Dil modelləri üçün bu, cümlədəki bir sözün mənasını müəyyənləşdirmək üçün digər sözlərə nə qədər əhəmiyyət verməli olduğunu öyrənmək deməkdir.

**Nümunə:** "Mən **kitabı** oxudum və o, çox maraqlı idi."
Burada "o" əvəzliyi "kitabı" sözünə aiddir. Model "o" sözünü emal edərkən, "kitabı" sözünə daha çox diqqət yetirməlidir. Diqqət mexanizmi məhz bu əlaqəni tapır.

## 14.2. Query, Key, Value (Soru, Açar, Dəyər)

Diqqət mexanizmi üç əsas matrisdən istifadə edir:

| Matris | Rolu | İzahı |
| :--- | :--- | :--- |
| **Query (Q)** | **Soru** | "Mən nə axtarıram?" (Məsələn, cari sözün təsviri). |
| **Key (K)** | **Açar** | "Məndə nə var?" (Məsələn, bütün digər sözlərin təsviri). |
| **Value (V)** | **Dəyər** | "Məlumat nədir?" (Məsələn, bütün digər sözlərin məlumatı). |

**İş Prinsipi:**
1.  **Uyğunluq Hesablanması:** Hər bir **Query** (Q) bütün **Key**-lər (K) ilə müqayisə edilir. Bu, hansı sözlərin cari sözlə əlaqəli olduğunu göstərən bir **Diqqət Balı (Attention Score)** yaradır.
2.  **Softmax:** Diqqət Balı **Softmax** funksiyasından keçirilərək **Diqqət Çəkisi (Attention Weight)**-ə çevrilir. Bu çəkilərin cəmi 1-ə bərabər olur.
3.  **Dəyərin Çəkilməsi:** Bu çəkilər **Value** (V) matrisi ilə vurulur. Nəticədə, model ən çox əlaqəli sözlərin məlumatını özündə cəmləşdirən yeni bir təsvir əldə edir.

## 14.3. Masked Self-Attention (Maskalanmış Öz-Diqqət)

Bizim GPT modelimiz **Generative (Yaradıcı)** modeldir. O, hər dəfə bir token yaradır və bu tokeni yaradarkən **yalnız özündən əvvəlki** tokenlərə baxa bilər.

*   **Self-Attention:** Model cümlədəki hər bir sözün digər sözlərə diqqət yetirməsidir.
*   **Masked:** Proqnozlaşdırma zamanı modelin gələcəkdəki tokenlərə "baxmasının" qarşısını almaq üçün **Diqqət Balı Matrisinin** yuxarı üçbucağı **sıfır** (və ya çox kiçik mənfi ədəd) ilə doldurulur.

Bu, modelin təlim prosesini daha çətin, lakin daha realistik edir.

## 14.4. Praktika: PyTorch-da Diqqət Mexanizmi

Gəlin, PyTorch-da sadə bir Diqqət Mexanizminin əsasını quraq.

**`attention.py`**

```python
import torch
import torch.nn as nn
from torch.nn import functional as F

# Modelin əsas hiperparametrləri (Gün 13-dən)
n_embd = 768  # Embedding ölçüsü
block_size = 256 # Kontekst uzunluğu

class Head(nn.Module):
    """Tək bir diqqət başı (Single Attention Head)"""
    
    def __init__(self, head_size):
        super().__init__()
        # Q, K, V matrislərini yaratmaq üçün xətti laylar
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        
        # Maskanı yaddaşda saxlamaq (buffer)
        # Bu, modelin gələcəyə baxmasının qarşısını alır
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
    def forward(self, x):
        # x-in ölçüsü: (Batch, Time, Channel) -> (B, T, C)
        B, T, C = x.shape
        
        # Q, K, V matrislərini hesablamaq
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        
        # 1. Diqqət Balını Hesablamaq (Q @ K.transpose)
        # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        wei = q @ k.transpose(-2, -1) * C**-0.5 # Skalalama
        
        # 2. Maskalanma (Masking)
        # Gələcək tokenlərə diqqət yetirməyin qarşısını alır
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        
        # 3. Softmax
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        
        # 4. Dəyərin Çəkilməsi (wei @ V)
        v = self.value(x) # (B, T, head_size)
        out = wei @ v     # (B, T, head_size)
        
        return out

# Nümunə: Tək bir diqqət başı yaratmaq
head = Head(head_size=n_embd // 12) # 768 / 12 = 64
print(head)
```

**Gündəlik Tapşırıq:** `attention.py` skriptini yaradın. Kodu oxuyun və **`wei = wei.masked_fill(...)`** sətrinin Masked Self-Attention-ı necə tətbiq etdiyini dərindən başa düşün.
