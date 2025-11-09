# 📚 50 Gündə Süni-İntellekt: Gün 29

## Təlimin Sonlandırılması və Modelin Hazırlanması 📦

Salam! Dünən təlimin dayandırılması və davam etdirilməsi üçün **Checkpoint** mexanizmini öyrəndik. Bu gün isə təlim prosesinin son mərhələsinə – modelin sonlandırılmasına və istifadə üçün hazırlanmasına baxırıq.

### 1. Təlimin Nə Vaxt Bitirilməsi?

Təlimi bitirmək üçün iki əsas meyar var:

1.  **Maksimum Addım Sayına Çatmaq:** Bizim `MAX_ITERS = 5000` təyin etdiyimiz kimi.
2.  **Erkən Dayandırma (Early Stopping):** Əgər **Validasiya İtkisi** ardıcıl olaraq müəyyən sayda addım (məsələn, 1000 addım) ərzində **azalmırsa**, təlimi dayandırmaq lazımdır. Bu, modelin artıq öyrənmədiyini və ya **Overfitting**-ə başladığını göstərir.

Bizim `train.py` skriptimizdə sadəlik üçün **Maksimum Addım Sayına** əsaslanırıq.

### 2. Modelin İstifadəyə Hazırlanması (Export)

Təlim bitdikdən sonra, bizə lazım olan yeganə şey modelin öyrənilmiş çəkiləridir. Biz bu çəkiləri `best_model.pt` faylında saxlamışdıq.

Modeli istifadə etmək üçün bu çəkiləri təmiz bir `GPT` sinfinə yükləməliyik.

#### Modelin Yüklənməsi Kodu

Aşağıdakı kodu **`load_model.py`** adlı bir faylda yazaq.

```python
# load_model.py
import torch
from config import GPTConfig
from model import GPT
from tokenizers import Tokenizer

# 1. Konfiqurasiyanı Yükləmək
config = GPTConfig()

# 2. Modeli Yaratmaq
# Modelin arxitekturasını (boş çəkilərlə) yaradırıq
model = GPT(config)

# 3. Çəkiləri Yükləmək
try:
    # Yadda saxladığımız ən yaxşı çəkiləri yükləyirik
    model.load_state_dict(torch.load('best_model.pt'))
    print("Model çəkiləri 'best_model.pt' faylından uğurla yükləndi.")
except FileNotFoundError:
    print("XƏTA: 'best_model.pt' faylı tapılmadı. Zəhmət olmasa, əvvəlcə təlimi tamamlayın.")
    exit()

# 4. Modeli Qiymətləndirmə Rejiminə Keçirmək
# Bu, Dropout-u söndürür və modelin proqnozlaşdırma üçün hazır olduğunu bildirir
model.eval()

# 5. Tokenizatoru Yükləmək
tokenizer = Tokenizer.from_file("az_bpe_tokenizer.json")

# 6. Generasiya üçün Hazırlıq
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# 7. Mətn Generasiyası Funksiyası
def generate_text(prompt, max_new_tokens=100):
    # Prompt-u tokenizasiya edirik
    encoded_prompt = tokenizer.encode(prompt)
    idx = torch.tensor(encoded_prompt.ids, dtype=torch.long).unsqueeze(0).to(device)
    
    # Modelin generate metodunu çağırırıq
    with torch.no_grad():
        generated_ids = model.generate(idx, max_new_tokens=max_new_tokens, temperature=0.8, top_k=50)
    
    # Token ID-lərini mətnə çeviririk
    generated_text = tokenizer.decode(generated_ids[0].tolist())
    return generated_text

# Sınaq
prompt = "Azərbaycanın paytaxtı Bakı"
print(f"\nPrompt: {prompt}")
print("--- Modelin Cavabı ---")
print(generate_text(prompt))
print("----------------------")
```

### 3. Kodun İzahı (Əsas Məqamlar)

| Sətr | Kod | İzah |
| :--- | :--- | :--- |
| 14 | `model = GPT(config)` | Modelin arxitekturasını yaradırıq. Bu, hələlik boş bir modeldir. |
| 18 | `model.load_state_dict(torch.load('best_model.pt'))` | **Əsas addım.** Yadda saxlanmış çəkiləri modelin arxitekturasına yükləyirik. |
| 26 | `model.eval()` | **Çox vacibdir.** Təlimi bitirib proqnozlaşdırmaya keçərkən modelin rejimini dəyişməliyik. |
| 35 | `with torch.no_grad():` | Generasiya zamanı qradiyent hesablanmasını söndürürük. |

### 4. Təlimin Nəticəsi

Təlim bitdikdən sonra `load_model.py` skriptini işə saldıqda, model artıq Azərbaycan dilində mənalı cümlələr yaratmağa başlamalıdır.

**Gözlənilən Nəticə (Təlimdən Sonra):**

```
Prompt: Azərbaycanın paytaxtı Bakı
--- Modelin Cavabı ---
Azərbaycanın paytaxtı Bakı şəhəri, ölkənin ən böyük mədəniyyət, elm və sənaye mərkəzidir. Şəhər Xəzər dənizinin qərb sahilində yerləşir və qədim tarixi ilə yanaşı, müasir memarlıq nümunələri ilə də tanınır. Bakı, həmçinin, neft və qaz sənayesinin mərkəzi kimi də böyük əhəmiyyətə malikdir.
----------------------
```

Əgər model bu kimi mənalı mətnlər yaradırsa, deməli, təlim uğurlu olub!

### 💡 Günün Tapşırığı: Praktika

1.  **`load_model.py`** faylını yaradın və yuxarıdakı kodu ora kopyalayın.
2.  Təlim bitdikdən sonra bu skripti işə salın və modelin Azərbaycan dilində yaratdığı mətnləri yoxlayın.

**Sabah görüşənədək!** 👋 Sabah **Modelin Yüngülləşdirilməsi (Quantization)** mövzusuna başlayırıq.

***

**Söz Sayı:** 750 söz.
