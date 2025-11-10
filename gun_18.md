# Gün 18: Parametr Sayının Hesablanması 🔢

## 18.1. Parametr Nədir?

**Parametrlər** modelin təlim zamanı öyrəndiyi dəyişənlərdir. Bu, modelin yaddaşı və biliyidir. Modelin nə qədər güclü olduğunu göstərən əsas göstəricilərdən biridir. Bizim hədəfimiz **100 Milyon (100M)** parametrdir.

Parametrlər əsasən **Xətti Laylarda (Linear Layers)** və **Embedding Cədvəllərində (Embedding Tables)** yerləşir.

## 18.2. Parametrlərin Hesablanması

Gəlin, Gün 17-də qurduğumuz `GPTModel` sinfinin parametr sayını hesablayaq.

**Modelin Əsas Hiperparametrləri:**
*   `vocab_size` (V): 32000
*   `n_embd` (C): 768
*   `n_layer` (L): 12
*   `n_head` (H): 12

### A. Embedding Layları

1.  **Token Embedding (`token_embedding_table`):**
    *   Hər bir token üçün `C` ölçülü vektor.
    *   Parametr Sayı: $V \times C = 32000 \times 768 = 24,576,000$

2.  **Position Embedding (`position_embedding_table`):**
    *   `block_size` (256) mövqe üçün `C` ölçülü vektor.
    *   Parametr Sayı: $256 \times 768 = 196,608$

### B. Transformer Blokları (12 ədəd)

Hər bir `Block` (Blok) aşağıdakılardan ibarətdir:

1.  **Multi-Head Attention (MHA):**
    *   **Q, K, V Layları:** Hər biri $C \times C$ ölçüdədir. $3 \times (C \times C)$
    *   **Proj Layı:** $C \times C$ ölçüdədir.
    *   **MHA-da Cəmi:** $4 \times (C \times C) = 4 \times (768 \times 768) = 2,359,296$

2.  **Feed-Forward Network (FFN):**
    *   **Lay 1:** $C \times (4C)$ ölçüdədir.
    *   **Lay 2:** $(4C) \times C$ ölçüdədir.
    *   **FFN-də Cəmi:** $2 \times (C \times 4C) = 8 \times C^2 = 8 \times (768 \times 768) = 4,718,592$

3.  **Layer Norm Layları:** Parametrləri çox azdır (təxminən $2 \times C$ hər lay üçün). Ümumi hesablamada nəzərə alınmır.

*   **Bir Blokda Cəmi:** $2,359,296 + 4,718,592 = 7,077,888$
*   **12 Blokda Cəmi:** $12 \times 7,077,888 = 84,934,656$

### C. Yekun Proqnozlaşdırma Başı

1.  **Linear Head (`lm_head`):**
    *   Parametr Sayı: $C \times V = 768 \times 32000 = 24,576,000$

### D. Ümumi Parametr Sayı

| Hissə | Parametr Sayı |
| :--- | :--- |
| Token Embedding | 24,576,000 |
| Position Embedding | 196,608 |
| 12 Transformer Bloku | 84,934,656 |
| Linear Head | 24,576,000 |
| **Ümumi Cəmi** | **134,283,264** |

**Nəticə:** Bizim modelimiz təxminən **134 Milyon** parametrə malikdir. Bu, sizin hədəflədiyiniz **100M** parametrə çox yaxındır və bu ölçü ilə **RTX 2050 (4GB VRAM)** üzərində təlim etmək mümkündür.

## 18.3. Praktika: PyTorch ilə Hesablama

PyTorch-da parametr sayını avtomatik hesablamaq üçün funksiya yazaq.

**`count_params.py`**

```python
import torch
# GPTModel sinfini (Gün 17-dən) bura kopyalayın və ya import edin

def count_parameters(model):
    """Modelin ümumi parametr sayını hesablayır."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params

# Modelin yaradılması
model = GPTModel()

total, trainable = count_parameters(model)

print(f"Ümumi Parametr Sayı: {total:,}")
print(f"Təlim Edilə Bilən Parametr Sayı: {trainable:,}")
print(f"Model Ölçüsü (Milyon): {total / 1_000_000:.2f} M")
```

**Gündəlik Tapşırıq:** `count_params.py` skriptini işə salın və hesablamalarımızın doğruluğunu yoxlayın. Bu, modelin ölçüsünü və VRAM tələbini anlamaq üçün vacibdir.
