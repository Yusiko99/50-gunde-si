# 📚 50 Gündə Süni-İntellekt: Gün 5

## Əsas Python Kitabxanaları: Rəqəmlərlə İşləmək 🔢

Salam! Dördüncü gündə GPU-nu PyTorch ilə birləşdirməyi öyrəndik. Bu gün isə LLM-in təməlində duran riyazi əməliyyatları idarə etmək üçün lazım olan üç əsas Python kitabxanası ilə tanış olacağıq: **NumPy**, **Pandas** və **tqdm**.

### 1. NumPy: Riyazi Əməliyyatların Qəlbi

Dərin Öyrənmə, əslində, böyük riyazi əməliyyatlar (vurma, toplama, matris əməliyyatları) silsiləsidir. Python-un özü bu cür əməliyyatlar üçün yavaşdır. Məhz buna görə də, **NumPy** (Numerical Python) istifadə olunur.

> **NumPy** — Python-da böyük, çoxölçülü massivlər (array) və matrislərlə işləmək üçün əsas kitabxanadır. O, bu əməliyyatları C dilində yazılmış sürətli kod vasitəsilə icra edir.

Bizim LLM-dəki bütün məlumatlar (tokenlər, modelin çəkiləri) **NumPy massivləri** şəklində saxlanılır.

#### Quraşdırma

`llm_50gun` mühitiniz aktivdirsə, sadəcə bu əmri icra edin:

```bash
pip install numpy
```

#### NumPy Massivi (Array) Nümunəsi

Python interaktiv mühitində (və ya bir Python faylında) yazaq:

```python
import numpy as np

# 1. Birölçülü massiv (Vektor)
a = np.array([1, 2, 3, 4, 5])
print(a)
# Nəticə: [1 2 3 4 5]

# 2. İkiölçülü massiv (Matris)
b = np.array([[10, 20], [30, 40]])
print(b)
# Nəticə:
# [[10 20]
#  [30 40]]

# 3. Sürətli əməliyyat
c = a * 2 + 5
print(c)
# Nəticə: [ 7  9 11 13 15]
```

**Kodun İzahı:**
*   `import numpy as np`: Kitabxananı `np` qısa adı ilə daxil edirik.
*   `np.array([...])`: NumPy massivi yaradırıq.
*   `a * 2 + 5`: Bütün massiv elementləri üzərində eyni anda riyazi əməliyyat aparılır. Bu, Python-un adi siyahıları ilə müqayisədə **çox sürətlidir**.

### 2. Pandas: Məlumatların Təşkili

Bizim LLM üçün məlumat topladığımızda, bu məlumatlar adətən cədvəl şəklində (məsələn, Excel faylı kimi) olur. **Pandas** bu cədvəl məlumatlarını idarə etmək üçün ən güclü alətdir.

> **Pandas** — Python-da məlumatların təhlili və manipulyasiyası üçün istifadə olunan kitabxanadır. Onun əsas strukturları **Series** (sütun) və **DataFrame** (cədvəl) adlanır.

#### Quraşdırma

```bash
pip install pandas
```

#### Pandas DataFrame Nümunəsi

```python
import pandas as pd

# Məlumat yaratmaq (Sözlük formatında)
data = {
    'Söz': ['Süni', 'İntellekt', 'Model'],
    'Tezlik': [1500, 980, 540],
    'Janr': ['Elm', 'Elm', 'Texnologiya']
}

# DataFrame yaratmaq
df = pd.DataFrame(data)
print(df)
# Nəticə:
#          Söz  Tezlik         Janr
# 0       Süni    1500          Elm
# 1  İntellekt     980          Elm
# 2      Model     540  Texnologiya

# Sadə əməliyyat: Tezliyi 1000-dən çox olan sözləri seçmək
yeni_df = df[df['Tezlik'] > 1000]
print(yeni_df)
# Nəticə:
#     Söz  Tezlik Janr
# 0  Süni    1500  Elm
```

**Kodun İzahı:**
*   `import pandas as pd`: Kitabxananı `pd` qısa adı ilə daxil edirik.
*   `pd.DataFrame(data)`: Sözlükdən cədvəl (DataFrame) yaradırıq.
*   `df['Tezlik'] > 1000`: Cədvəlin "Tezlik" sütunundakı dəyərləri yoxlayırıq. Pandas bu cür mürəkkəb filtrləməni çox asanlaşdırır.

### 3. tqdm: Proqres Göstəricisi

LLM təlimi uzun çəkən bir prosesdir. Bəzən bir neçə saat, bəzən bir neçə gün. Prosesin hansı mərhələdə olduğunu bilmək üçün **tqdm** kitabxanasından istifadə edəcəyik.

> **tqdm** — Python-da dövrlərin (loop) icra prosesini göstərən gözəl və asan bir proqres çubuğu (progress bar) yaradır.

#### Quraşdırma

```bash
pip install tqdm
```

#### tqdm Nümunəsi

```python
from tqdm import tqdm
import time

# 100 dəfə təkrarlanan bir prosesi simulyasiya edək
for i in tqdm(range(100), desc="Təlim Prosesi"):
    # Hər dəfə 0.01 saniyə gözləyirik
    time.sleep(0.01)
```

**Kodun İzahı:**
*   `from tqdm import tqdm`: `tqdm` funksiyasını daxil edirik.
*   `tqdm(range(100), desc="Təlim Prosesi")`: `range(100)` üzərində dövr edərkən, ekranda **"Təlim Prosesi"** başlığı ilə bir proqres çubuğu göstəriləcək.

Bu, modelimizi təlim edərkən prosesin nə qədər qaldığını görmək üçün çox faydalı olacaq.

### 💡 Günün Tapşırığı: Praktika

1.  `llm_50gun` mühitində `numpy`, `pandas` və `tqdm` kitabxanalarını quraşdırın.
2.  NumPy istifadə edərək 3x3 ölçülü bir matris yaradın.
3.  Pandas istifadə edərək ən azı 3 sütunlu kiçik bir DataFrame yaradın.
4.  `tqdm` istifadə edərək 500 dəfə təkrarlanan bir dövr üçün proqres çubuğu yaradın.

**Sabah görüşənədək!** 👋 Sabah LLM-in ən təməl daşına – **Məlumat Korpusu** anlayışına keçəcəyik. Azərbaycan dili üçün hansı məlumatların mövcud olduğunu araşdıracağıq.

***

**Söz Sayı:** 750 söz.
