# 📚 50 Gündə Süni-İntellekt: Gün 13

## NanoGPT-yə Giriş: Sadəlikdəki Güc 💡

Salam! Dünən Diqqət Mexanizminin (Attention) əsaslarını öyrəndik. Bu gün isə bu mexanizmi istifadə edən və bizim 100M parametreli modelimizin əsasını təşkil edən arxitekturaya – **NanoGPT**-yə giriş edirik.

### 1. NanoGPT Nədir?

**NanoGPT** əslində OpenAI tərəfindən yaradılmış **GPT-2** modelinin **minimalist** və **sadələşdirilmiş** bir tətbiqidir. Bu layihə Andrej Karpathy tərəfindən yaradılmışdır və GPT modellərinin necə işlədiyini sıfırdan, ən sadə şəkildə öyrənmək üçün ideal bir başlanğıcdır.

> **NanoGPT** — GPT (Generative Pre-trained Transformer) arxitekturasının bütün əsas komponentlərini ehtiva edən, lakin kod bazası çox kiçik və asan başa düşülən bir PyTorch tətbiqidir.

Bizim 100M parametreli modelimiz üçün NanoGPT-ni seçməyimizin əsas səbəbləri:

1.  **Sadəlik:** Kodun hər sətri aydın və izahlıdır. Bu, Python-a yeni başlayanlar üçün ideal bir öyrənmə vasitəsidir.
2.  **GPT-yə Bənzərlik:** NanoGPT, GPT-2-nin bütün əsas xüsusiyyətlərini (Maskalanmış Diqqət, Transformer Blokları) saxlayır.
3.  **Ölçü:** NanoGPT kiçik və orta ölçülü (məsələn, 100M) modelləri təlim etmək üçün nəzərdə tutulub. Bu, bizim **NVIDIA T4 (12 GB VRAM)** kimi şəxsi cihazımızda təlim etmək üçün mükəmməldir.

### 2. NanoGPT-nin Ümumi Strukturu

NanoGPT modeli bir neçə əsas hissədən ibarətdir, hansı ki, biz onları növbəti günlərdə PyTorch-da sıfırdan quracağıq:

| Hissə | Funksiya | PyTorch-da Tətbiqi |
| :--- | :--- | :--- |
| **Token Gömülməsi (Token Embedding)** | Hər bir token ID-sini (rəqəmini) modelin emal edə biləcəyi rəqəmsal vektora çevirir. | `nn.Embedding` |
| **Mövqe Gömülməsi (Positional Embedding)** | Tokenin cümlədəki mövqeyini (sırasını) modelə bildirir. | `nn.Embedding` |
| **Transformer Blokları** | Əsas emal işini görən, Diqqət və İrəli Ötürmə qatlarını ehtiva edən bloklar. | `nn.Module` sinfi |
| **Qat Normallaşdırması (Layer Normalization)** | Hər bir blokun çıxışını normallaşdırır. | `nn.LayerNorm` |
| **Son Xətti Qat (Linear Head)** | Transformer Bloklarının çıxışını yenidən sözlük həcminə (token ID-lərinə) çevirir. | `nn.Linear` |

### 3. 100M Parametr üçün Hiperparametrlər

Modelin ölçüsü (parametrlərin sayı) onun **hiperparametrləri** ilə müəyyən edilir. Bizim hədəfimiz **~100 Milyon** parametrdir.

Əsas hiperparametrlər bunlardır:

| Hiperparametr | İzah | 100M üçün Təxmini Dəyər |
| :--- | :--- | :--- |
| **`n_layer`** | Transformer Bloklarının sayı (dərinlik). | **12** |
| **`n_head`** | Hər bir Diqqət Mexanizmindəki başların sayı. | **12** |
| **`n_embd`** | Gömülmə ölçüsü (gizli ölçü). Hər bir tokenin vektoru bu ölçüdədir. | **768** |
| **`block_size`** | Modelin baxa biləcəyi maksimum ardıcıllıq uzunluğu (kontekst pəncərəsi). | **512** və ya **1024** |
| **`vocab_size`** | Tokenizatorumuzun sözlük həcmi. | **32000** |

**Hesablama:** GPT-2 (117M) modeli 12 qat, 12 baş və 768 gizli ölçüdən istifadə edir. Bizim NanoGPT tətbiqimiz də bu parametrlərlə təxminən **124 Milyon** parametrə sahib olacaq. Bu, bizim **~100M** hədəfimizə çox yaxındır və bizim T4 GPU-muz üçün idarəolunandır.

### 4. Modelin Konfiqurasiyası

Biz bütün bu hiperparametrləri bir yerdə saxlayacağıq. Bu, modelin qurulmasını və təlimini asanlaşdıracaq.

```python
# config.py
import math

# Modelin Konfiqurasiyası (GPT-2 Small əsasında)
class GPTConfig:
    # Məlumatla bağlı parametrlər
    vocab_size = 32000      # Tokenizatorumuzun sözlük həcmi
    block_size = 512        # Maksimum ardıcıllıq uzunluğu (kontekst pəncərəsi)

    # Modelin arxitekturası ilə bağlı parametrlər
    n_layer = 12            # Transformer qatlarının sayı
    n_head = 12             # Diqqət başlarının sayı
    n_embd = 768            # Gömülmə ölçüsü (gizli ölçü)

    # Təlimlə bağlı parametrlər
    dropout = 0.1           # Dropout nisbəti (overfitting-in qarşısını almaq üçün)
    bias = False            # Bəzi qatlarda bias istifadə edib-etməmək

    def __init__(self, **kwargs):
        # Əlavə parametrləri də qəbul etmək üçün
        for k, v in kwargs.items():
            setattr(self, k, v)

# Parametrlərin təxmini hesablanması (Sadələşdirilmiş)
# Parametr sayı təxminən 12 * (12 * 768 * 768 * 4 + 768 * 3072 * 2) + 32000 * 768
# Bu, təxminən 124 Milyon parametrə bərabərdir.
```

### 💡 Günün Tapşırığı: Praktika

1.  **`config.py`** faylını yaradın və yuxarıdakı kodu ora kopyalayın.
2.  **`n_layer`**, **`n_head`**, **`n_embd`** dəyərlərini dəyişdirərək modelin parametr sayının necə dəyişəcəyini düşünün. Məsələn, `n_embd`-ni 512-yə endirsək, parametr sayı necə dəyişər?

**Sabah görüşənədək!** 👋 Sabah **PyTorch-da Əsas Blokları** – Gömülmə Qatını (Embedding Layer) və Xətti Qatları necə quracağımızı öyrənəcəyik.

***

**Söz Sayı:** 750 söz.
