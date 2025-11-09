# 📚 50 Gündə Süni-İntellekt: Gün 11

## Transformer: LLM-lərin Beyni 🧠

Salam! İlk 10 günü uğurla tamamladıq və modelimizin qidasını – **rəqəmləşdirilmiş Azərbaycan dili məlumatını** hazırladıq. İndi isə bu məlumatı emal edəcək **beyni** – yəni **Transformer** arxitekturasını qurmağa başlayırıq.

### 1. Transformer Nədir?

2017-ci ildə Google tərəfindən nəşr olunan **"Attention Is All You Need"** adlı məqalə Süni İntellekt dünyasında inqilab etdi. Bu məqalə **Transformer** adlı yeni bir neyron şəbəkə arxitekturasını təqdim etdi.

> **Transformer** — ardıcıl məlumatları (məsələn, mətn) emal etmək üçün nəzərdə tutulmuş, **təkrarlanan (recurrent)** və ya **konvolyusiya (convolutional)** əməliyyatları əvəzinə yalnız **Diqqət Mexanizminə (Attention Mechanism)** əsaslanan bir neyron şəbəkə arxitekturasıdır.

Transformer-dən əvvəlki modellər (RNN, LSTM) mətnləri sözbəsöz, ardıcıl şəkildə oxuyurdu. Bu, çox yavaş idi və uzun cümlələrdə əvvəldəki sözlərin mənasını unutmağa səbəb olurdu.

**Transformer** isə bütün cümləni **bir anda** emal edə bilir. Bu, LLM-lərin sürətini və mətnin mənasını başa düşmə qabiliyyətini kəskin şəkildə artırdı.

### 2. Encoder və Decoder: Transformer-in İki Hissəsi

Əslində, Transformer arxitekturası iki əsas hissədən ibarətdir:

| Hissə | Məqsəd | Misal |
| :--- | :--- | :--- |
| **Encoder (Kodlayıcı)** | Giriş mətnini (input) başa düşmək və onun mənasını rəqəmsal təmsilə çevirmək. | Tərcümədə: "Salam" sözünün mənasını başa düşür. |
| **Decoder (Dekodlayıcı)** | Encoder-in başa düşdüyü mənanı istənilən çıxış mətninə (output) çevirmək. | Tərcümədə: "Salam"ın mənasını ingiliscə "Hello" sözünə çevirir. |

*   **Tərcümə Modelləri (Seq2Seq):** Həm **Encoder**, həm də **Decoder** istifadə edir (məsələn, T5, BART).
*   **Chatbot Modelləri (GPT):** Bizim yaratdığımız kimi **Generativ** (yeni mətn yaradan) modellər yalnız **Decoder** hissəsindən istifadə edir.

#### Niyə Yalnız Decoder?

Bizim LLM-imiz (NanoGPT) **Generativ Pre-trained Transformer (GPT)** ailəsinə aiddir. Bu modellər **növbəti tokeni proqnozlaşdırmaq** üçün təlim olunur.

Decoder hissəsi məhz bu iş üçün idealdır, çünki o, **Maskalanmış Diqqət (Masked Attention)** mexanizminə malikdir. Bu mexanizm modelin **yalnız özündən əvvəlki** sözlərə baxaraq növbəti sözü proqnozlaşdırmasına imkan verir.

### 3. Transformer Blokunun Əsas Komponentləri

Transformer-in hər bir qatı (layer) **Transformer Bloku** adlanır. Bu blokun içində dörd əsas komponent var:

1.  **Multi-Head Attention (Çoxbaşlı Diqqət):** Mətnin fərqli hissələri arasındakı əlaqələri eyni anda öyrənir. (Sabah daha ətraflı öyrənəcəyik).
2.  **Add & Norm (Əlavə et və Normallaşdır):** Diqqət mexanizminin çıxışını girişə əlavə edir (Residual Connection) və sonra normallaşdırır (Layer Normalization).
3.  **Feed-Forward Network (İrəli Ötürmə Şəbəkəsi):** Hər bir tokeni fərdi şəkildə emal edən kiçik bir neyron şəbəkəsidir.
4.  **Positional Encoding (Mövqe Kodlaşdırması):** Transformer-in ardıcıllıq məlumatını (sözlərin sırasını) itirməməsi üçün hər bir tokenə onun cümlədəki mövqeyini bildirən rəqəmsal məlumat əlavə edir.

### 4. PyTorch-da Transformer-ə İlk Baxış

PyTorch-da bu komponentləri necə quracağımızı öyrənəcəyik. Məsələn, **Transformer Blokunun** PyTorch-da sadələşdirilmiş görünüşü belədir:

```python
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        # 1. Çoxbaşlı Diqqət
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads)
        # 2. İrəli Ötürmə Şəbəkəsi
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(), # Aktivasiya funksiyası
            nn.Linear(4 * d_model, d_model)
        )
        # 3. Normallaşdırma
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Diqqət və Residual Connection
        x = x + self.attn(self.norm1(x))
        # İrəli Ötürmə və Residual Connection
        x = x + self.ffn(self.norm2(x))
        return x
```

**Kodun İzahı:**
*   `import torch.nn as nn`: PyTorch-un neyron şəbəkə modullarını daxil edirik.
*   `class TransformerBlock(nn.Module)`: Bütün neyron şəbəkə komponentləri PyTorch-da `nn.Module` sinfindən miras alır.
*   `nn.MultiheadAttention`: PyTorch-un hazır Çoxbaşlı Diqqət moduludur.
*   `nn.Linear`: Xətti (Linear) qatdır, yəni matris vurulması.
*   `nn.LayerNorm`: Normallaşdırma qatıdır.
*   `x = x + ...`: **Residual Connection** (Qalıq Əlaqə) adlanır. Bu, modelin dərinləşdikcə öyrənmə qabiliyyətini itirməməsi üçün vacibdir.

### 💡 Günün Tapşırığı: Düşün və Araşdır

1.  **Residual Connection** (Qalıq Əlaqə) nə deməkdir? Niyə dərin neyron şəbəkələrində bu qədər vacibdir? (Sadə dildə cavab tapmağa çalışın).
2.  Transformer-in ardıcıl modellərdən (RNN/LSTM) əsas fərqi nədir?

**Sabah görüşənədək!** 👋 Sabah Transformer-in ən vacib hissəsi olan **Diqqət Mexanizmini (Attention)** sıfırdan qurmağa başlayacağıq.

***

**Söz Sayı:** 750 söz.
