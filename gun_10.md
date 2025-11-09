# 📚 50 Gündə Süni-İntellekt: Gün 10

## Məlumatın Hazırlanması: Təlimə Son Hazırlıq 🎯

Salam! İlk 10 günlük mərhələmizin sonuna çatdıq! Dünən **Azərbaycan dili üçün xüsusi BPE Tokenizatorumuzu** uğurla təlim etdik. Bu gün isə bu tokenizatoru istifadə edərək bütün məlumatımızı modelin təlimi üçün son formaya gətirəcəyik.

### 1. Məlumatın Təlimə Hazırlanması Nədir?

Modelimiz mətnləri deyil, **rəqəmlər ardıcıllığını** qəbul edir. Hazırlıq prosesi iki əsas addımdan ibarətdir:

1.  **Tokenizasiya:** Bütün `azcorpus_cleaned.txt` faylını tokenizatorumuz vasitəsilə rəqəmlər ardıcıllığına çevirmək.
2.  **Təlim/Validasiya Bölünməsi:** Məlumatın bir hissəsini **Təlim (Train)** üçün (modelin öyrənəcəyi hissə), digər hissəsini isə **Validasiya (Validation)** üçün (modelin öyrənmədiyini yoxlamaq üçün) ayırmaq.

### 2. Məlumatın Hazırlanması Kodu

Aşağıdakı kodu **`prepare_data.py`** adlı bir faylda yazaq.

```python
# prepare_data.py
import os
import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm

# 1. Əsas Parametrlər
TOKENIZER_FILE = "az_bpe_tokenizer.json"
INPUT_FILE = "azcorpus_cleaned.txt"
# Məlumatın nə qədərinin validasiya üçün ayrılacağı (5%)
VALIDATION_SPLIT = 0.05

# 2. Tokenizatoru Yükləmək
print(f"1. Tokenizator '{TOKENIZER_FILE}' yüklənir...")
try:
    tokenizer = Tokenizer.from_file(TOKENIZER_FILE)
except Exception as e:
    print(f"XƏTA: Tokenizator faylı tapılmadı və ya yüklənmədi: {e}")
    print("Zəhmət olmasa, əvvəlcə 'train_tokenizer.py' skriptini icra edin.")
    exit()

# 3. Məlumatı Oxumaq
print(f"2. Məlumat '{INPUT_FILE}' oxunur...")
with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    # Bütün mətnləri bir böyük sətir kimi oxuyuruq
    data = f.read()

# 4. Məlumatı Tokenizasiya Etmək
print("3. Məlumat tokenizasiya edilir...")
# Tokenizatorun 'encode' metodu mətni rəqəmlər ardıcıllığına çevirir
encoding = tokenizer.encode(data)
token_ids = np.array(encoding.ids, dtype=np.uint16) # uint16 yaddaşa qənaət edir

print(f"   Ümumi token sayı: {len(token_ids):,}")
print(f"   Yaddaşda tutduğu yer: {token_ids.nbytes / (1024*1024):.2f} MB")

# 5. Təlim və Validasiya Bölünməsi
# Məlumatı təlim və validasiya hissələrinə ayırırıq
split_point = int(len(token_ids) * (1 - VALIDATION_SPLIT))

train_data = token_ids[:split_point]
val_data = token_ids[split_point:]

print(f"4. Məlumat bölündü (Validasiya nisbəti: {VALIDATION_SPLIT*100}%)")
print(f"   Təlim token sayı: {len(train_data):,}")
print(f"   Validasiya token sayı: {len(val_data):,}")

# 6. NumPy formatında Yadda Saxlamaq
# Token ID-lərini gələcəkdə PyTorch-da asanlıqla yükləmək üçün .npy formatında saxlayırıq
np.save('train.npy', train_data)
np.save('val.npy', val_data)

print("\n5. Hazırdır! 'train.npy' və 'val.npy' faylları yaradıldı.")
print("Modelin təlimi üçün məlumat bazası tam hazırdır!")
```

### 3. Kodun İzahı (Hər Sətrin Detallı İzahı)

| Sətr | Kod | İzah |
| :--- | :--- | :--- |
| 4 | `import os, numpy as np, ...` | Lazım olan kitabxanaları daxil edirik. `numpy` rəqəmlər ardıcıllığını effektiv idarə etmək üçün vacibdir. |
| 10 | `VALIDATION_SPLIT = 0.05` | Məlumatın **5%-ni** validasiya üçün ayırırıq. Bu, standart bir nisbətdir. |
| 15 | `tokenizer = Tokenizer.from_file(TOKENIZER_FILE)` | Dünən yaratdığımız **`az_bpe_tokenizer.json`** faylını yükləyirik. |
| 23 | `data = f.read()` | Bütün mətn faylını (azcorpus) bir sətir kimi oxuyuruq. |
| 27 | `encoding = tokenizer.encode(data)` | Bütün mətn sətirini tokenizatorumuz vasitəsilə rəqəmlər ardıcıllığına çeviririk. |
| 28 | `token_ids = np.array(encoding.ids, dtype=np.uint16)` | Çıxan rəqəmlər siyahısını **NumPy massivinə** çeviririk. `np.uint16` istifadə edirik, çünki 32000 sözlük həcmi üçün 16 bit (65535-ə qədər rəqəm) kifayətdir və yaddaşa qənaət edir. |
| 33 | `split_point = int(len(token_ids) * (1 - VALIDATION_SPLIT))` | 95% təlim, 5% validasiya olacaq şəkildə kəsmə nöqtəsini hesablayırıq. |
| 35 | `train_data = token_ids[:split_point]` | Kəsmə nöqtəsinə qədər olan hissəni təlim məlumatı kimi ayırırıq. |
| 36 | `val_data = token_ids[split_point:]` | Kəsmə nöqtəsindən sonrakı hissəni validasiya məlumatı kimi ayırırıq. |
| 43 | `np.save('train.npy', train_data)` | Təlim məlumatını **.npy** formatında yadda saxlayırıq. Bu, NumPy massivlərini sürətli yükləmək üçün standart formadır. |
| 44 | `np.save('val.npy', val_data)` | Validasiya məlumatını yadda saxlayırıq. |

### 4. İcra

`llm_50gun` mühitiniz aktivdirsə, kodu icra edin:

```bash
python prepare_data.py
```

Nəticədə, iki böyük fayl yaranacaq: **`train.npy`** və **`val.npy`**. Bu fayllar bizim modelimizin təlimi üçün lazım olan bütün rəqəmləşdirilmiş Azərbaycan dili mətnlərini ehtiva edir.

### 💡 Günün Tapşırığı: Praktika

1.  `prepare_data.py` faylını yaradın və icra edin.
2.  Yaranan **`train.npy`** və **`val.npy`** fayllarının ölçülərini yoxlayın. (Məlumatın həcmindən asılı olaraq bir neçə yüz meqabayt ola bilər).
3.  **Təbrik edirəm!** İlk 10 günlük mərhələni tamamladınız. Artıq LLM-in təməli hazırdır. Sabah **Transformer** arxitekturasına keçirik!

**Sabah görüşənədək!** 👋

***

**Söz Sayı:** 750 söz.
