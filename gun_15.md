# 📚 50 Gündə Süni-İntellekt: Gün 15

## Çoxbaşlı Diqqət (Multi-Head Attention) 👁️‍🗨️

Salam! Dünən NanoGPT modelimizin təməl qatlarını (Embedding, Linear) qurduq. Bu gün isə modelin ən güclü və mürəkkəb hissəsinə – **Çoxbaşlı Diqqət (Multi-Head Attention)** mexanizminə keçirik.

### 1. Niyə "Çoxbaşlı"?

Dünən öyrəndiyimiz **Self-Attention** mexanizmi bir sözün cümlədəki digər sözlərlə olan əlaqəsini tapır. Lakin bu, yalnız **bir növ** əlaqəni tapır.

Məsələn, "Azərbaycanın **gözəl** paytaxtı **Bakı**dır." cümləsində:
*   Bir diqqət başı "**gözəl**" sözünün "**Bakı**" sözü ilə əlaqəsini (sifət-isim əlaqəsi) tapa bilər.
*   Başqa bir diqqət başı isə "**Azərbaycanın**" sözünün "**paytaxtı**" sözü ilə əlaqəsini (yiyəlik-mənsubiyyət əlaqəsi) tapa bilər.

> **Çoxbaşlı Diqqət** — eyni anda bir neçə (bizim halımızda **12**) fərqli diqqət mexanizmini paralel şəkildə işlətmək deməkdir. Hər bir "baş" mətnin fərqli bir aspektinə, fərqli bir əlaqə növünə fokuslanır.

Bu, modelin mətnin bütün incəliklərini, qrammatik və semantik əlaqələrini eyni anda öyrənməsinə imkan verir.

### 2. Çoxbaşlı Diqqətin İş Prinsipi

1.  **Bölünmə:** Giriş vektoru (`n_embd=768`) **`n_head=12`** sayda bərabər hissəyə bölünür. Hər bir hissənin ölçüsü `768 / 12 = 64` olur.
2.  **Paralel Hesablama:** Hər bir kiçik hissə üzərində müstəqil olaraq **Self-Attention** (Maskalanmış) əməliyyatı aparılır.
3.  **Birləşdirmə:** Bütün 12 başın çıxışları (hər biri 64 ölçülü) yenidən birləşdirilir və əvvəlki ölçüyə (`768`) qaytarılır.
4.  **Son Xətti Qat:** Birləşdirilmiş çıxış son bir xətti qatdan keçirilir.

### 3. PyTorch-da Çoxbaşlı Diqqətin Qurulması

Biz dünənki **SelfAttention** sinfini **MultiHeadAttention** sinfinin içində istifadə edəcəyik.

Aşağıdakı kodu **`attention.py`** faylına əlavə edək (və ya yenidən yazaq).

```python
# attention.py (Davamı)
import torch
import torch.nn as nn
from torch.nn import functional as F
from config import GPTConfig

# Dünənki SelfAttention sinfi (sadəlik üçün burada təkrar yazılmır, amma ehtiyac var)
# ... (SelfAttention sinfi buraya əlavə olunmalıdır) ...

class MultiHeadAttention(nn.Module):
    """ Çoxbaşlı Maskalanmış Öz-Diqqət Mexanizmi """

    def __init__(self, config):
        super().__init__()
        # Modelin hiperparametrlərini konfiqurasiyadan alırıq
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.head_size = self.n_embd // self.n_head # Hər başın ölçüsü: 768 / 12 = 64

        # Bütün Q, K, V proyeksiyalarını eyni anda edən xətti qat
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=config.bias)
        # Birləşdirilmiş çıxışı emal edən son xətti qat
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

        # Maskanı yaratmaq (yalnız bir dəfə)
        # Bu, modelin özündən sonrakı tokenlərə baxmasının qarşısını alır
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # Batch, Ardıcıllıq Uzunluğu, Gömülmə Ölçüsü (768)

        # 1. Q, K, V-ni Hesablamaq
        # c_attn(x) -> (B, T, 3 * C)
        # split(3) -> (B, T, C), (B, T, C), (B, T, C)
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # 2. Çoxbaşlı Şəklə Salmaq
        # (B, T, C) -> (B, T, n_head, head_size) -> (B, n_head, T, head_size)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # 3. Scaled Dot-Product Attention
        # (B, n_head, T, head_size) @ (B, n_head, head_size, T) -> (B, n_head, T, T)
        att = (q @ k.transpose(-2, -1)) * (self.head_size**-0.5)

        # 4. Maskalanma (Masking)
        # Özündən sonrakı tokenlərə diqqəti sıfıra endiririk
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))

        # 5. Softmax və Dropout
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # 6. Dəyərin Çəkilməsi (wei @ V)
        # out -> (B, n_head, T, head_size)
        out = att @ v

        # 7. Başları Birləşdirmək
        # (B, n_head, T, head_size) -> (B, T, n_head, head_size) -> (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # 8. Son Proyeksiya
        out = self.resid_dropout(self.c_proj(out))
        return out
```

### 4. Kodun İzahı (Əsas Məqamlar)

| Sətr | Kod | İzah |
| :--- | :--- | :--- |
| 23 | `self.head_size = self.n_embd // self.n_head` | Hər başın ölçüsünü hesablayır (768 / 12 = 64). |
| 26 | `self.c_attn = nn.Linear(..., 3 * self.n_embd, ...)` | **Əsas fərq budur!** Q, K, V-ni ayrı-ayrı 3 xətti qatdan keçirmək əvəzinə, bir böyük qatdan keçirib sonra 3 bərabər hissəyə bölürük. Bu, daha səmərəlidir. |
| 35 | `q, k, v = self.c_attn(x).split(self.n_embd, dim=2)` | Girişdən çıxan 3 * 768 ölçülü vektoru Q, K, V (hər biri 768 ölçülü) olaraq bölürük. |
| 39 | `k = k.view(...).transpose(1, 2)` | Vektoru **Çoxbaşlı** formata salırıq: `(B, T, 12, 64)` -> `(B, 12, T, 64)`. İndi 12 baş paralel işləyə bilər. |
| 44 | `att = (q @ k.transpose(-2, -1)) * ...` | Diqqət çəkilərini hesablayırıq. |
| 53 | `out = att @ v` | Diqqət çəkilərini dəyərlərə tətbiq edirik. |
| 57 | `out = out.transpose(1, 2).contiguous().view(B, T, C)` | 12 başın çıxışını yenidən birləşdirib əvvəlki `(B, T, 768)` formasına qaytarırıq. |
| 60 | `out = self.resid_dropout(self.c_proj(out))` | Son xətti qatdan keçirib **Dropout** tətbiq edirik.

### 💡 Günün Tapşırığı: Düşün və Praktika

1.  **`attention.py`** faylını yaradın və `MultiHeadAttention` sinfini ora kopyalayın.
2.  Niyə Q, K, V-ni ayrı-ayrı qatlardan keçirmək əvəzinə, bir böyük qatdan keçirib bölmək daha səmərəlidir? (Cavab: GPU-lar böyük matris əməliyyatlarını kiçik əməliyyatlardan daha sürətli icra edir).

**Sabah görüşənədək!** 👋 Sabah **Transformer Blokunun** bütün komponentlərini (Çoxbaşlı Diqqət, LayerNorm, Feed-Forward) birləşdirəcəyik.

***

**Söz Sayı:** 750 söz.
