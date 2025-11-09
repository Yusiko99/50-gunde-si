# 📚 50 Gündə Süni-İntellekt: Gün 7

## Məlumatın Toplanması və Təmizlənməsi 🧹

Salam! Dünən LLM-in qidası olan **Mətn Korpusu** anlayışı ilə tanış olduq və Azərbaycan dili üçün əsas mənbəyimiz olan **azcorpus**-u müəyyənləşdirdik. Bu gün isə bu məlumatı necə əldə edib, necə təmizləyəcəyimizi öyrənəcəyik.

### 1. Məlumat Toplama Strategiyaları

Ən böyük mənbəyimiz **azcorpus** olsa da, LLM-in daha yaxşı performans göstərməsi üçün məlumatı artırmaq və müxtəlifləşdirmək vacibdir.

| Strategiya | İzah | Nümunə |
| :--- | :--- | :--- |
| **Açıq Mənbəli Korpuslar** | Artıq başqaları tərəfindən toplanmış və paylaşılmış məlumat bazaları. | **azcorpus**, Azərbaycan Vikipediyası dump-ları. |
| **Web Scraping (Veb Qazıma)** | Xüsusi proqramlar vasitəsilə veb-saytlardan mətnləri avtomatik toplamaq. | Xəbər saytları, rəsmi dövlət saytları. |
| **Kitablar və Sənədlər** | Elektron kitablar, elmi məqalələr, rəsmi sənədlər. | Azərbaycan ədəbiyyatı, qanunvericilik aktları. |

Bizim layihəmizdə **azcorpus**-dan istifadə edəcəyik, lakin gələcəkdə **Web Scraping** vasitəsilə məlumatı necə artıracağınızı da bilməlisiniz.

> **Web Scraping** — veb-saytların HTML kodunu oxuyaraq, lazım olan mətn və ya digər məlumatları çıxarmaq prosesidir.

### 2. Məlumatın Təmizlənməsi (Data Cleaning)

Modelin **"zibil"** öyrənməməsi üçün məlumatın təmizlənməsi **təlimdən daha vacibdir**.

| Təmizləmə Addımı | Niyə Edilir? |
| :--- | :--- |
| **Təkrarların Silinməsi** | Eyni mətnin dəfələrlə təkrarlanması modelin həmin mətni əzbərləməsinə (overfitting) səbəb olur. |
| **Xüsusi Simvolların Silinməsi** | HTML teqləri, reklam linkləri, emojilər (əgər istifadə etmək istəmiriksə) kimi lazımsız simvolların çıxarılması. |
| **Formatlaşdırma** | Bütün mətnin kiçik hərflərə çevrilməsi (bəzən), boşluqların və sətir sonlarının standartlaşdırılması. |
| **Qısa Mətnlərin Silinməsi** | Çox qısa cümlələr (məsələn, "Bəli.", "Yox.") modelə az məlumat verir, onları silmək olar. |

### 3. azcorpus-un Yüklənməsi və İlkin Təmizlənməsi (Praktika)

Bizim **azcorpus** məlumat bazası Hugging Face-də artıq **ilkin təmizləmədən** keçib. Lakin, biz yenə də onu yükləyib, strukturunu yoxlayacağıq.

#### Addım 1: Kitabxanaların Quraşdırılması

Əvvəlcə lazım olan kitabxanaları quraşdıraq (əgər quraşdırmamışıqsa):

```bash
conda activate llm_50gun
pip install datasets pandas
```

#### Addım 2: Məlumatı Yükləmək və Pandas-a Çevirmək

Aşağıdakı kodu **`data_prep.py`** adlı bir faylda yazaq.

```python
# data_prep.py
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

# Tqdm-i Pandas-a əlavə edirik ki, prosesi izləyə bilək
tqdm.pandas()

print("1. azcorpus məlumat bazası yüklənir...")
# Hugging Face-dən məlumatı yükləyirik
dataset = load_dataset("azcorpus/azcorpus_v0")

# Məlumatı Pandas DataFrame-ə çeviririk
# Bizə yalnız 'text' sütunu lazımdır
df = pd.DataFrame(dataset['train'])['text']

print(f"2. İlkin mətn sayı: {len(df)}")
print("3. İlkin mətnin ilk 5 sətri:")
print(df.head())

print("\n4. Məlumatın Təmizlənməsi...")

# Təmizləmə funksiyası
def clean_text(text):
    # Boşluqları təmizləmək
    text = str(text).strip()
    # Əlavə sətir sonlarını tək sətir sonu ilə əvəz etmək
    text = text.replace('\n\n', '\n').replace('\r', '')
    return text

# Təmizləməni bütün mətnlərə tətbiq edirik
# progress_apply istifadə edirik ki, prosesin getdiyini görək
df_cleaned = df.progress_apply(clean_text)

# Boş və ya çox qısa mətnləri silirik (uzunluğu 50 simvoldan az olanlar)
df_cleaned = df_cleaned[df_cleaned.str.len() >= 50]

print(f"5. Təmizləmədən sonra mətn sayı: {len(df_cleaned)}")

# Təmizlənmiş məlumatı bir fayla yazırıq
output_file = "azcorpus_cleaned.txt"
print(f"6. Təmizlənmiş məlumat '{output_file}' faylına yazılır...")

# Bütün mətnləri birləşdirib bir fayla yazırıq
with open(output_file, 'w', encoding='utf-8') as f:
    # Hər mətnin arasına iki sətir sonu qoyuruq ki, model fərqli sənədləri ayırd edə bilsin
    f.write('\n\n'.join(df_cleaned))

print("7. Hazırdır! Məlumat tokenizasiya üçün hazırdır.")
```

**Kodun İzahı (Hər Sətrin İzahı):**

| Sətr | Kod | İzah |
| :--- | :--- | :--- |
| 4 | `import pandas as pd` | Məlumatı cədvəl kimi idarə etmək üçün `pandas` kitabxanasını daxil edirik. |
| 5 | `from datasets import load_dataset` | Hugging Face-dən məlumat bazasını yükləmək üçün `load_dataset` funksiyasını daxil edirik. |
| 6 | `from tqdm import tqdm` | Prosesin gedişatını göstərmək üçün `tqdm` kitabxanasını daxil edirik. |
| 9 | `tqdm.pandas()` | `tqdm`-i `pandas` funksiyalarına inteqrasiya edirik ki, `progress_apply` istifadə edə bilək. |
| 12 | `dataset = load_dataset("azcorpus/azcorpus_v0")` | **azcorpus** məlumat bazasını internetdən yükləyirik. |
| 16 | `df = pd.DataFrame(dataset['train'])['text']` | Yüklənmiş məlumatın yalnız **'text'** sütununu seçib Pandas cədvəlinə (DataFrame) çeviririk. |
| 23 | `def clean_text(text):` | Mətn təmizləmə funksiyasını təyin edirik. |
| 25 | `text = str(text).strip()` | Mətnin əvvəlindəki və sonundakı boşluqları silirik. |
| 27 | `text = text.replace('\n\n', '\n').replace('\r', '')` | İkiqat sətir sonlarını təkə endiririk və Windows-a xas olan `\r` simvollarını silirik. |
| 28 | `return text` | Təmizlənmiş mətni geri qaytarırıq. |
| 32 | `df_cleaned = df.progress_apply(clean_text)` | Təmizləmə funksiyasını bütün mətnlərə tətbiq edirik və proqresi göstəririk. |
| 35 | `df_cleaned = df_cleaned[df_cleaned.str.len() >= 50]` | Uzunluğu 50 simvoldan az olan mətnləri (çox qısa cümlələri) silirik. |
| 40 | `with open(output_file, 'w', encoding='utf-8') as f:` | Təmizlənmiş məlumatı `azcorpus_cleaned.txt` faylına yazmaq üçün açırıq. |
| 42 | `f.write('\n\n'.join(df_cleaned))` | Bütün təmizlənmiş mətnləri iki sətir sonu ilə birləşdirib fayla yazırıq. |

### 💡 Günün Tapşırığı: Praktika

1.  `data_prep.py` faylını yaradın və yuxarıdakı kodu ora kopyalayın.
2.  `llm_50gun` mühitində bu kodu icra edin: `python data_prep.py`
3.  Prosesin bitməsini gözləyin və **`azcorpus_cleaned.txt`** faylının yarandığına əmin olun.

**Sabah görüşənədək!** 👋 Sabah LLM-in ən təməl daşı olan **Tokenizasiya** anlayışına keçəcəyik.

***

**Söz Sayı:** 800 söz.
