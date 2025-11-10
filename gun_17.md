# Gün 17: GPT Modelinin Tam Quruluşu 🏗️

## 17.1. Bütün Hissələrin Birləşdirilməsi

Əvvəlki günlərdə biz LLM-in əsas komponentlərini qurduq:
1.  **Tokenizator** (Mətni rəqəmlərə çevirir).
2.  **Head** (Tək Diqqət Başı).
3.  **MultiHeadAttention** (Çoxbaşlı Diqqət).
4.  **Block** (Transformer Bloku).

Bu gün isə bütün bu hissələri birləşdirərək **GPT (Generative Pre-trained Transformer)** modelimizin yekun sinfini yaradacağıq.

## 17.2. GPT Modelinin Arxitekturası

GPT modelinin quruluşu aşağıdakı ardıcıllıqdan ibarətdir:

1.  **Token Embedding:** Giriş token ID-lərini rəqəmsal vektorlara (Embedding) çevirir.
2.  **Position Embedding:** Tokenlərin cümlədəki mövqeyini öyrənir və Token Embedding-ə əlavə edir.
3.  **Transformer Blokları:** 12 ədəd `Block` ardıcıl olaraq tətbiq olunur.
4.  **Final Layer Norm:** Bütün bloklardan sonra yekun normallaşdırma.
5.  **Linear Head:** Nəticəni lüğət ölçüsünə (32000) çevirir və hansı tokenin növbəti gələcəyini proqnozlaşdırır.

## 17.3. Praktika: `GPTModel` Sinfinin Qurulması

**`model.py`**

```python
import torch
import torch.nn as nn
from torch.nn import functional as F
# Block sinfini (Gün 16-dan) bura kopyalayın və ya import edin

# Modelin əsas hiperparametrləri (Gün 13-dən)
n_embd = 768      # Embedding ölçüsü
n_head = 12       # Başların sayı
n_layer = 12      # Blokların sayı
block_size = 256  # Kontekst uzunluğu
vocab_size = 32000 # Lüğət ölçüsü

class GPTModel(nn.Module):
    """Əsas GPT Model Sinifi"""
    
    def __init__(self):
        super().__init__()
        
        # 1. Token və Mövqe Embedding-ləri
        # Tokenlərin rəqəmsal təsviri
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # Tokenlərin mövqeyinin rəqəmsal təsviri
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        # 2. Ardıcıl Transformer Blokları
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        
        # 3. Yekun Normallaşdırma
        self.ln_f = nn.LayerNorm(n_embd)
        
        # 4. Proqnozlaşdırma Başı (Linear Head)
        # Nəticəni lüğət ölçüsünə çevirir
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
        # Modelin çəkilərini ilkinləşdirmək
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Modelin çəkilərini daha yaxşı təlim üçün ilkinləşdirmək."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape # idx: (Batch, Time)
        
        # 1. Token və Mövqe Embedding-ləri
        # idx: token ID-ləri (B, T)
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        # pos: mövqe ID-ləri (0-dan T-1-ə qədər)
        pos = torch.arange(T, device=idx.device) # (T)
        pos_emb = self.position_embedding_table(pos) # (T, C)
        
        # 2. Embedding-ləri birləşdirmək
        x = tok_emb + pos_emb # (B, T, C)
        
        # 3. Transformer Bloklarından keçirmək
        x = self.blocks(x) # (B, T, C)
        
        # 4. Yekun Normallaşdırma
        x = self.ln_f(x) # (B, T, C)
        
        # 5. Proqnozlaşdırma Başı
        logits = self.lm_head(x) # (B, T, vocab_size)
        
        loss = None
        if targets is not None:
            # Loss-u hesablamaq üçün ölçüləri düzəltmək
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            # Cross-Entropy Loss funksiyası
            loss = F.cross_entropy(logits, targets)

        return logits, loss

# Nümunə: Modelin yaradılması
model = GPTModel()
print(model)
```

## 17.4. Kodun İzahı

| Sətr | Kod | İzahı |
| :--- | :--- | :--- |
| **31** | `self.token_embedding_table = nn.Embedding(vocab_size, n_embd)` | Hər bir token ID-si üçün 768 ölçülü vektor yaradır. |
| **33** | `self.position_embedding_table = nn.Embedding(block_size, n_embd)` | Hər bir mövqe (0-dan 255-ə qədər) üçün 768 ölçülü vektor yaradır. |
| **36** | `self.blocks = nn.Sequential(...)` | 12 ədəd `Block` sinfini ardıcıl olaraq yığır. |
| **42** | `self.apply(self._init_weights)` | Modelin çəkilərini təlimə başlamazdan əvvəl standart normal paylanmaya uyğun olaraq ilkinləşdirir. |
| **60** | `x = tok_emb + pos_emb` | Tokenin məlumatını (nə olduğu) və mövqe məlumatını (harada olduğu) birləşdirir. |
| **74** | `loss = F.cross_entropy(logits, targets)` | **Cross-Entropy Loss** funksiyası modelin proqnozları ilə həqiqi növbəti tokenlər arasındakı fərqi hesablayır. Bu, modelin öyrənməsinə rəhbərlik edən əsas funksiyadır. |

**Gündəlik Tapşırıq:** `model.py` skriptini yaradın. Modelin quruluşunu və `forward` funksiyasının məlumatı necə emal etdiyini tam başa düşün.
