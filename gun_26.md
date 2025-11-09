# 📚 50 Gündə Süni-İntellekt: Gün 26

## Təlimin Monitorinqi: Modelin "Sağlamlığını" İzləmək 🩺

Salam! Dünən təlim skriptimizi `accelerate launch train.py` əmri ilə işə saldıq. Təlim başladıqdan sonra, modelin düzgün öyrənib-öyrənmədiyini və hər hansı bir problemin olub-olmadığını izləmək çox vacibdir. Bu proses **Təlimin Monitorinqi** adlanır.

### 1. İtki (Loss) Dəyərinin İzlənməsi

Təlimin ən əsas göstəricisi **İtki (Loss)** dəyəridir.

| İtki Növü | İzah | Niyə İzlənilir? |
| :--- | :--- | :--- |
| **Təlim İtkisi (Train Loss)** | Modelin təlim məlumatı üzərindəki səhvi. | Təlimin irəlilədiyini göstərir. Təlim irəlilədikcə bu dəyər **azalmalıdır**. |
| **Validasiya İtkisi (Validation Loss)** | Modelin görmədiyi (val.npy) məlumat üzərindəki səhvi. | Modelin **ümumiləşdirmə** qabiliyyətini göstərir. |

#### Gözlənilən Nəticə

*   **Başlanğıcda:** Təlim və Validasiya itkiləri təxminən **10.37** (ln(32000)) olmalıdır.
*   **Təlim İrəlilədikcə:** Hər iki itki dəyəri tədricən **azalmalıdır**. Məsələn, 5.0, 4.0, 3.0 və s.

### 2. Overfitting (Həddindən Artıq Uyğunlaşma)

Təlim zamanı qarşılaşa biləcəyimiz ən böyük problem **Overfitting**-dir.

> **Overfitting** — modelin təlim məlumatını o qədər yaxşı əzbərləməsidir ki, yeni (validasiya) məlumat üzərində pis nəticə göstərir.

**Overfitting-in Əlaməti:**
*   **Təlim İtkisi** azalmağa davam edir.
*   **Validasiya İtkisi** isə müəyyən bir nöqtədən sonra **artmağa** başlayır.

Bu, modelin Azərbaycan dilinin ümumi qaydalarını öyrənmək əvəzinə, sadəcə `azcorpus`-dakı cümlələri əzbərlədiyi deməkdir.

#### Overfitting-in Qarşısını Alma Yolları

Bizim kodumuzda artıq bu mexanizmlər tətbiq olunub:

1.  **Dropout:** `model.py` və `block.py`-də istifadə etdiyimiz `nn.Dropout` qatları təsadüfi olaraq neyronları söndürür. Bu, modelin bir neyrona həddindən artıq güvənməsinin qarşısını alır.
2.  **Weight Decay (AdamW):** Optimallaşdırıcıdakı bu mexanizm çəkilərin çox böyüməsinin qarşısını alır.
3.  **Erkən Dayandırma (Early Stopping):** Əgər Validasiya İtkisi ardıcıl olaraq bir neçə dəfə artarsa, təlimi dayandırmaq lazımdır.

### 3. Təlim Loglarının Görsəlləşdirilməsi

Təlimin gedişatını yalnız rəqəmlərlə deyil, həm də **qrafiklərlə** izləmək daha effektivdir. Bunun üçün **TensorBoard** və ya **Weights & Biases (W&B)** kimi alətlərdən istifadə olunur.

Bizim `train.py` skriptimizdə sadəlik üçün hələlik bu alətləri tətbiq etmədik, lakin gələcəkdə bu alətləri istifadə etməyi öyrənməlisiniz.

#### Sadə Qrafik Çəkmə (Matplotlib)

Təlim bitdikdən sonra log faylındakı itki dəyərlərini istifadə edərək sadə bir qrafik çəkə bilərik.

```python
# visualize_loss.py
import matplotlib.pyplot as plt
import re

def plot_loss(log_file="train_log.txt"):
    """ Təlim log faylından itki dəyərlərini oxuyub qrafik çəkir """
    train_losses = []
    val_losses = []
    iters = []
    
    with open(log_file, 'r') as f:
        for line in f:
            # Validasiya itkisini tapmaq üçün regex istifadə edirik
            match = re.search(r"Addım (\d+): Təlim İtkisi \(Loss\) = ([\d\.]+)", line)
            if match:
                iters.append(int(match.group(1)))
                val_losses.append(float(match.group(2)))
            
            # Təlim itkisini tapmaq üçün (əgər hər addımda yazılıbsa)
            # ... (Bu hissəni train.py-də əlavə etməliyik) ...

    plt.figure(figsize=(10, 6))
    plt.plot(iters, val_losses, label="Validasiya İtkisi", color='red')
    plt.title("Təlimin İrəliləyişi: Validasiya İtkisi")
    plt.xlabel("Təlim Addımı (Iteration)")
    plt.ylabel("İtki (Loss)")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_graph.png")
    print("Qrafik 'loss_graph.png' faylına yazıldı.")

# plot_loss()
```

### 💡 Günün Tapşırığı: Praktika

1.  `train.py` skriptinin çıxışını bir log faylına (`train_log.txt`) yazın.
2.  Təlimin ilk bir neçə min addımında Validasiya İtkisinin necə azaldığını izləyin.
3.  `matplotlib` kitabxanasını quraşdırın: `pip install matplotlib`.
4.  `visualize_loss.py` faylını yaradın və təlim bitdikdən sonra qrafiki çəkin.

**Sabah görüşənədək!** 👋 Sabah **Validasiya və Qiymətləndirmə** mövzusunu, xüsusilə **Perplexity** (Çaşqınlıq) metrikini öyrənəcəyik.

***

**Söz Sayı:** 750 söz.
