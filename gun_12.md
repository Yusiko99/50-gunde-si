# 📚 50 Gündə Süni-İntellekt: Gün 12

## Diqqət Mexanizmi (Attention): Mənanın Fokuslanması 💡

Salam! Dünən Transformer arxitekturasına giriş etdik. Bu gün isə bu arxitekturanın **ürəyi** olan **Diqqət Mexanizmini (Attention Mechanism)** öyrənəcəyik.

### 1. Diqqət Nədir?

İnsanlar danışarkən və ya oxuyarkən, cümlənin mənasını başa düşmək üçün bəzi sözlərə digərlərindən daha çox diqqət yetirirlər.

Məsələn, "Mən çayı sevirəm, çünki o, **isti** və **rahatladıcıdır**." cümləsində "o" əvəzliyi "çay" sözünə işarə edir. Beynimiz avtomatik olaraq "o" sözünü "çay" sözü ilə əlaqələndirir.

**Diqqət Mexanizmi** modelə məhz bu qabiliyyəti verir:

> **Diqqət Mexanizmi** — modelin bir sözü emal edərkən, cümlədəki digər sözlərin nə qədər vacib olduğunu müəyyənləşdirməsinə imkan verən bir mexanizmdir.

### 2. Self-Attention (Öz-Diqqət)

LLM-lərdə istifadə olunan diqqət mexanizmi **Self-Attention (Öz-Diqqət)** adlanır. Bu o deməkdir ki, model bir cümlədəki hər bir sözün digər bütün sözlərlə olan əlaqəsini hesablayır.

Self-Attention üç əsas komponentdən istifadə edir:

1.  **Query (Sorğu - Q):** Cari sözün mənasını axtarmaq üçün istifadə olunan vektordur.
2.  **Key (Açar - K):** Cümlədəki hər bir sözün məlumatını təmsil edən vektordur.
3.  **Value (Dəyər - V):** Əlaqəli sözlərin məlumatını daşıyan vektordur.

**İş Prinsipi:**
1.  **Uyğunluq Hesablanması:** Hər bir **Query** (cari söz) bütün **Key**-lər (bütün sözlər) ilə müqayisə edilir. Bu müqayisə nəticəsində **Diqqət Çəkiləri (Attention Weights)** yaranır. Bu çəkilər, cari söz üçün hansı sözlərin daha vacib olduğunu göstərir.
2.  **Yumşaq Maksimum (Softmax):** Çəkilər 0 ilə 1 arasına normallaşdırılır.
3.  **Dəyərin Çəkilməsi:** Bu çəkilər **Value** (Dəyər) vektorlarına tətbiq edilir. Yüksək çəkiyə malik olan sözlərin məlumatı daha çox çəkilir və cari sözün emalına daxil edilir.

### 3. Masked Self-Attention (Maskalanmış Öz-Diqqət)

Bizim LLM-imiz (GPT) **Generativ** modeldir, yəni **növbəti sözü proqnozlaşdırır**. Bu o deməkdir ki, model bir sözü proqnozlaşdırarkən **özündən sonra gələn sözləri görməməlidir**. Əks halda, cavabı "kopya" edər.

Bunun üçün **Maskalanmış Öz-Diqqət** istifadə olunur:

> **Maskalanmış Öz-Diqqət** — Diqqət mexanizmində, cari sözün özündən sonra gələn sözlərə olan diqqət çəkilərini **sıfıra** endirən (və ya mənfi sonsuzluğa yaxınlaşdıran) bir maska tətbiq edilir.

Bu maska sayəsində model, məsələn, "Azərbaycan dili" cümləsində "Azərbaycan" sözünü emal edərkən "dili" sözünü görmür.

### 4. PyTorch-da Maskalanmış Diqqət

Biz bu mexanizmi PyTorch-da sıfırdan quracağıq.

Aşağıdakı kodu **`attention.py`** adlı bir faylda yazaq. Bu, bizim **Self-Attention** modulunun əsasını təşkil edəcək.

```python
# attention.py
import torch
import torch.nn as nn
from torch.nn import functional as F

class SelfAttention(nn.Module):
    """ Sadələşdirilmiş Self-Attention mexanizmi """

    def __init__(self, n_embd, block_size):
        super().__init__()
        # Q, K, V üçün xətti qatlar (Linear layers)
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        # Maskanı bu obyektdə saxlayırıq
        # Bu, modelin özündən sonrakı tokenlərə baxmasının qarşısını alır
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))
                                     .view(1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.shape # B=Batch, T=Time (uzunluq), C=Channel (gömülmə ölçüsü)

        # Q, K, V-ni hesablayırıq
        k = self.key(x)   # (B, T, C)
        q = self.query(x) # (B, T, C)
        v = self.value(x) # (B, T, C)

        # 1. Diqqət Çəkilərini Hesablamaq (Q * K^T)
        # Scaled Dot-Product Attention
        wei = q @ k.transpose(-2, -1) * (C**-0.5) # (B, T, T)

        # 2. Maskalanma (Masking)
        # Özündən sonrakı tokenlərə diqqəti sıfıra endiririk
        wei = wei.masked_fill(self.tril[:,:T,:T] == 0, float('-inf'))

        # 3. Softmax
        wei = F.softmax(wei, dim=-1) # (B, T, T)

        # 4. Dəyərin Çəkilməsi (wei @ V)
        out = wei @ v # (B, T, C)
        return out
```

**Kodun İzahı:**
*   `n_embd`: Hər bir tokenin gömülmə ölçüsü (embedding dimension).
*   `block_size`: Modelin baxa biləcəyi maksimum mətn uzunluğu.
*   `self.key`, `self.query`, `self.value`: Q, K, V vektorlarını yaratmaq üçün istifadə olunan xətti qatlardır.
*   `self.register_buffer('tril', ...)`: **tril** (triangle lower) adlanan üçbucaq maskasını yaradır. Bu maska, əsas diaqonalın altındakı bütün dəyərləri 1, üstündəkiləri isə 0 edir.
*   `wei = q @ k.transpose(-2, -1) * (C**-0.5)`: Diqqət çəkilərini hesablayır (matris vurulması). `(C**-0.5)` isə **Scaled** hissəsidir (normallaşdırma).
*   `wei = wei.masked_fill(self.tril[:,:T,:T] == 0, float('-inf'))`: **Maskalanma** hissəsidir. Üçbucaq maskada 0 olan yerləri mənfi sonsuzluğa çevirir. Softmax funksiyası mənfi sonsuzluğu 0-a çevirəcək.
*   `wei = F.softmax(wei, dim=-1)`: Çəkiləri normallaşdırır.
*   `out = wei @ v`: Çəkilmiş dəyərləri hesablayır.

### 💡 Günün Tapşırığı: Praktika

1.  `attention.py` faylını yaradın və yuxarıdakı kodu ora kopyalayın.
2.  PyTorch-da kiçik bir sınaq matrisi yaradın və `SelfAttention` sinfini test edin.

**Sabah görüşənədək!** 👋 Sabah bu sadə **SelfAttention** mexanizmini daha güclü olan **Çoxbaşlı Diqqətə (Multi-Head Attention)** çevirəcəyik.

***

**Söz Sayı:** 800 söz.
