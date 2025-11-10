# Gün 12: Məlumatın Hazırlanması: Rəqəmləşdirmə 🧱

## 12.1. Təlim Məlumatının Forması

LLM-lər **Növbəti Tokenin Proqnozlaşdırılması (Next Token Prediction)** tapşırığı üzərində təlim keçir. Bu o deməkdir ki, modelə verilən hər hansı bir token ardıcıllığı üçün, model bu ardıcıllıqdan sonra gəlmə ehtimalı ən yüksək olan tokeni proqnozlaşdırmalıdır.

**Məntiq:** Təlim məlumatı **giriş (input)** və **hədəf (target)** ardıcıllıqlarına bölünməlidir.

| Token Ardıcıllığı | Giriş (X) | Hədəf (Y) |
| :--- | :--- | :--- |
| **T1 T2 T3 T4 T5** | T1 T2 T3 T4 | T2 T3 T4 T5 |

Model T1-ə baxıb T2-ni, T1 və T2-yə baxıb T3-ü proqnozlaşdırmağı öyrənir.

## 12.2. Məlumatın Bloklara Bölünməsi

LLM-lər yalnız müəyyən bir uzunluğa qədər olan ardıcıllıqları emal edə bilər. Bu uzunluq **Kontekst Pəncərəsi (Context Window)** və ya **Blok Ölçüsü (`block_size`)** adlanır. Bizim modelimiz üçün bu ölçü **256 token** olaraq təyin edilmişdir.

**Məntiq:** Korpusdakı bütün mətn, 256 tokenlik bloklara bölünməlidir.

### A. Təkrar Yükləmə (Overlapping)

Korpusu bloklara bölərkən, məlumat itkisinin qarşısını almaq üçün **təkrar yükləmə (overlapping)** texnikası istifadə olunur.

*   **Sadə Bölmə:** `[T1..T256]`, `[T257..T512]`
*   **Təkrar Yükləmə:** `[T1..T256]`, `[T129..T384]`, `[T257..T512]`

Bu, modelin bir cümlənin ortasında kəsilməsi səbəbindən konteksti itirməsinin qarşısını alır.

## 12.3. Praktika: Məlumatın Rəqəmləşdirilməsi

**`prepare_data.py`**

```python
from tokenizers import Tokenizer
import numpy as np
import torch
import os

TOKENIZER_FILE = "az_llm-tokenizer.json"
CORPUS_FILE = "normalized_corpus.txt"
BLOCK_SIZE = 256 # Modelin kontekst pəncərəsi

def prepare_data():
    """Korpusu tokenizasiya edir və 256 tokenlik bloklara bölür."""
    
    # 1. Tokenizatoru Yükləmək
    tokenizer = Tokenizer.from_file(TOKENIZER_FILE)
    
    # 2. Korpusu Oxumaq
    with open(CORPUS_FILE, 'r', encoding='utf-8') as f:
        text = f.read()
        
    # 3. Bütün Korpusu Tokenizasiya Etmək
    # Tokenizator bütün mətni bir dəfəyə token ID-lərinə çevirir.
    encoded = tokenizer.encode(text)
    all_token_ids = encoded.ids
    
    # 4. Təlim və Validasiya Məlumatına Bölmək
    # Məlumatın 90%-i təlim, 10%-i validasiya üçün istifadə olunur.
    data = torch.tensor(all_token_ids, dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    # 5. Bloklara Bölmə (Təkrar Yükləməsiz Sadə Versiya)
    # Təkrar yükləmə mürəkkəb olduğu üçün, sadəlik üçün ardıcıl bloklara bölürük.
    
    # Təlim məlumatını bloklara bölmək
    train_blocks = []
    for i in range(0, len(train_data) - BLOCK_SIZE + 1, BLOCK_SIZE):
        train_blocks.append(train_data[i:i + BLOCK_SIZE])
        
    # Validasiya məlumatını bloklara bölmək
    val_blocks = []
    for i in range(0, len(val_data) - BLOCK_SIZE + 1, BLOCK_SIZE):
        val_blocks.append(val_data[i:i + BLOCK_SIZE])
        
    # 6. Yekun Tensorları Yadda Saxlamaq
    train_tensor = torch.stack(train_blocks)
    val_tensor = torch.stack(val_blocks)
    
    torch.save(train_tensor, 'train_data.pt')
    torch.save(val_tensor, 'val_data.pt')
    
    print(f"Məlumat hazırlığı tamamlandı.")
    print(f"Təlim bloklarının sayı: {train_tensor.shape[0]}")
    print(f"Validasiya bloklarının sayı: {val_tensor.shape[0]}")

if __name__ == "__main__":
    if not os.path.exists(TOKENIZER_FILE):
        print("Xəta: Tokenizator faylı tapılmadı. Zəhmət olmasa Gün 11-i tamamlayın.")
    else:
        prepare_data()
```

## 12.4. Kodun Məntiqi İzahı

| Sətr | Kod | Məntiqi İzahı |
| :--- | :--- | :--- |
| **24** | `data = torch.tensor(all_token_ids, dtype=torch.long)` | Bütün token ID-lərini PyTorch-un `Long` tipli tensoruna çevirir. `Long` tipi tam ədədləri saxlamaq üçün istifadə olunur. |
| **25** | `n = int(0.9 * len(data))` | **Məntiq:** Məlumatın 90%-i modelin öyrənməsi üçün (Təlim), 10%-i isə modelin öyrənmədiyini yoxlamaq üçün (Validasiya) ayrılır. Bu, modelin **Overfitting** (həddindən artıq əzbərləmə) edib-etmədiyini yoxlamağa kömək edir. |
| **32** | `range(0, len(train_data) - BLOCK_SIZE + 1, BLOCK_SIZE)` | **Məntiq:** Korpusu ardıcıl olaraq 256 tokenlik hissələrə bölür. `+ 1` son blokun tam 256 token olmasını təmin edir. |
| **41** | `torch.stack(train_blocks)` | Bütün 256 tokenlik blokları bir böyük tensor şəklində birləşdirir. Bu, `(Blokların Sayı, BLOCK_SIZE)` ölçüsündə bir matris yaradır. |
