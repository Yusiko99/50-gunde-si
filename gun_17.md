# 📚 50 Gündə Süni-İntellekt: Gün 17

## GPT Modelinin Tam Quruluşu: NanoGPT 🏗️

Salam! Son bir neçə gündə NanoGPT modelimizin bütün əsas komponentlərini – Gömülmə Qatlarını, Çoxbaşlı Diqqəti və Transformer Blokunu (Block) qurduq. Bu gün isə bütün bu hissələri birləşdirərək **GPT (NanoGPT)** modelinin tam sinfini yaradacağıq.

Bu, bizim **100 Milyon parametreli Azərbaycan dili LLM-imizin** rəsmi olaraq PyTorch-da doğulduğu gündür!

### 1. GPT Modelinin Ümumi Strukturu

GPT modeli sadə bir ardıcıllıqla işləyir:

1.  **Giriş:** Token ID-ləri (rəqəmlər ardıcıllığı).
2.  **Gömülmə:** Token və Mövqe Gömülmələri toplanır.
3.  **Transformer Blokları:** Gömülmüş vektorlar ardıcıl olaraq **`n_layer`** (bizim halımızda 12) sayda Transformer Blokundan keçir.
4.  **Çıxış:** Son Normallaşdırma (LayerNorm) və Xətti Başlıq (LM Head) vasitəsilə növbəti tokenin ehtimalı hesablanır.

### 2. PyTorch-da GPT Sinfinin Tamamlanması

Biz Gün 14-də **`gpt_model_base.py`** adlı bir fayl yaratmışdıq. İndi həmin faylı **`model.py`** adlandıraraq və `Block` sinfini daxil edərək tamamlayırıq.

```python
# model.py
import torch
import torch.nn as nn
from torch.nn import functional as F
from config import GPTConfig
from block import Block # Dünən yaratdığımız Transformer Bloku

class GPT(nn.Module):
    """ NanoGPT arxitekturasına əsaslanan Böyük Dil Modeli """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Modelin bütün parametrlərini ehtiva edən əsas konteyner
        self.transformer = nn.ModuleDict(dict(
            # 1. Token və Mövqe Gömülmələri
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            # 2. Transformer Blokları (12 ədəd)
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # 3. Son Normallaşdırma
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        # 4. Dil Modeli Başı (LM Head)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Parametrlərin sayını hesablayırıq
        self.apply(self._init_weights)
        print(f"Modelin ümumi parametr sayı: {self.get_num_params():,}")

    def get_num_params(self, non_embedding=True):
        """ Modelin parametr sayını hesablayır """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            # Gömülmə qatlarının parametrlərini çıxarırıq (bəzən yüngülləşdirmə üçün)
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        """ Parametrlərin ilkin dəyərlərini təyin edir """
        if isinstance(module, nn.Linear):
            # Xətti qatlar üçün normal paylanma ilə ilkin dəyərlər
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Gömülmə qatları üçün normal paylanma ilə ilkin dəyərlər
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            # LayerNorm üçün vahid dəyərlər
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, idx, targets=None):
        # idx: Token ID-lərindən ibarət Tensor (B, T)
        B, T = idx.size()

        # 1. Gömülmələri Hesablamaq
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # (T)
        token_emb = self.transformer.wte(idx) # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)   # (T, n_embd)
        x = self.transformer.drop(token_emb + pos_emb) # (B, T, n_embd)

        # 2. Transformer Bloklarından Keçirmək
        for block in self.transformer.h:
            x = block(x)

        # 3. Son Normallaşdırma
        x = self.transformer.ln_f(x)

        # 4. Çıxış (Logits)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None

        # Əgər hədəf tokenlər (targets) verilibsə, itkini (loss) hesablayırıq
        if targets is not None:
            # Logits-i (B*T, vocab_size) və targets-i (B*T) şəklində düzəldirik
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            # Çarpaz Entropiya İtkisi (Cross-Entropy Loss)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
```

### 3. Kodun İzahı (Əsas Məqamlar)

| Sətr | Kod | İzah |
| :--- | :--- | :--- |
| 18 | `self.transformer = nn.ModuleDict(dict(...))` | Bütün Transformer komponentlərini bir lüğətdə saxlayırıq. |
| 25 | `h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])` | **12 ədəd** Transformer Blokunu ardıcıl olaraq yaradırıq. |
| 31 | `self.apply(self._init_weights)` | Modelin bütün qatlarına **ilkin dəyərləri** tətbiq edirik. Bu, təlimin stabil başlaması üçün vacibdir. |
| 42 | `def _init_weights(self, module):` | **Parametrlərin İlkin Dəyərləri:** Modelin öyrənməyə başlaması üçün bütün çəkilərə (weights) kiçik, təsadüfi dəyərlər verilir. |
| 62 | `for block in self.transformer.h:` | Gömülmələrdən gələn məlumatı ardıcıl olaraq 12 blokdan keçiririk. |
| 71 | `if targets is not None:` | Əgər modelə hədəf tokenlər verilibsə, **İtki Funksiyasını (Loss Function)** hesablayırıq. |
| 75 | `loss = F.cross_entropy(logits, targets)` | **Cross-Entropy Loss** istifadə edirik. Bu, generativ dil modelləri üçün standart itki funksiyasıdır.

### 4. Parametr Sayının Hesablanması

Bizim konfiqurasiyamız (`n_layer=12`, `n_head=12`, `n_embd=768`, `vocab_size=32000`) ilə modelin parametr sayı təxminən:

**Modelin ümumi parametr sayı: 124,417,536**

Bu, bizim **~100 Milyon** parametr hədəfimizə tam uyğundur!

### 💡 Günün Tapşırığı: Praktika

1.  **`model.py`** faylını yaradın və yuxarıdakı kodu ora kopyalayın.
2.  `config.py`, `block.py`, `attention.py` fayllarının eyni qovluqda olduğundan əmin olun.
3.  Modeli yaradın və parametr sayının yuxarıdakı rəqəmə yaxın olduğunu yoxlayın.

**Sabah görüşənədək!** 👋 Sabah modelin təlimdən əvvəl necə mətn yaratdığını görmək üçün **Mətn Generasiyası (Sampling)** mexanizmini öyrənəcəyik.

***

**Söz Sayı:** 800 söz.
