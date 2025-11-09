# 📚 50 Gündə Süni-İntellekt: Gün 9

## Tokenizatorun Qurulması (Praktika) 🛠️

Salam! Dünən **Tokenizasiya** və **BPE** alqoritminin nəzəriyyəsini öyrəndik. Bu gün isə praktikaya keçirik və **azcorpus** məlumatımız üzərində **Azərbaycan dili üçün xüsusi BPE Tokenizatorumuzu** təlim edəcəyik.

Bu, bizim LLM layihəmizdə ilk dəfə **real kod** yazacağımız və icra edəcəyimiz gündür.

### 1. Tokenizatorun Təlimi üçün Kod

Aşağıdakı kodu **`train_tokenizer.py`** adlı bir faylda yazaq.

```python
# train_tokenizer.py
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# 1. Əsas Parametrlər
# Modelin tanıya biləcəyi unikal tokenlərin sayı
VOCAB_SIZE = 32000
# Təmizlənmiş mətn faylımızın yolu
FILE_PATH = "azcorpus_cleaned.txt"
# Tokenizatoru saxlayacağımız fayl adı
OUTPUT_FILE = "az_bpe_tokenizer.json"

# 2. Xüsusi Tokenlər
# Bu tokenlər model üçün xüsusi məna daşıyır
SPECIAL_TOKENS = [
    "<|endoftext|>", # Mətnin sonunu bildirir (GPT modelləri üçün vacibdir)
    "<|pad|>",       # Mətnləri eyni uzunluğa gətirmək üçün istifadə olunur
    "<|unk|>",       # Modelin tanımadığı tokenlər üçün
]

print(f"Tokenizator təliminə başlanılır. Sözlük həcmi: {VOCAB_SIZE}")

# 3. Tokenizatorun Yaradılması
# BPE modelini istifadə edən boş bir Tokenizer obyekti yaradırıq
tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))

# 4. Əvvəlcədən Tokenizasiya (Pre-Tokenization)
# Mətni boşluqlara görə ilkin tokenlərə ayırır
tokenizer.pre_tokenizer = Whitespace()

# 5. Təlimçi (Trainer) Obyektinin Yaradılması
# BPE alqoritmini məlumatımız üzərində təlim edəcək obyektdir
trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=SPECIAL_TOKENS,
    # Tokenlərin birləşməsinin minimum tezliyi (çox az təkrarlananları nəzərə almamaq üçün)
    min_frequency=2
)

# 6. Tokenizatorun Təlim Edilməsi
# Təlimçi, göstərilən faylı oxuyur və BPE alqoritmini tətbiq edir
tokenizer.train([FILE_PATH], trainer)

# 7. Tokenizatorun Saxlanması
# Təlim olunmuş tokenizatoru JSON formatında yadda saxlayırıq
tokenizer.save(OUTPUT_FILE)

print(f"Tokenizator uğurla təlim edildi və '{OUTPUT_FILE}' faylına yazıldı.")

# 8. Yoxlama (Test)
# Saxlanmış tokenizatoru yükləyib sınaqdan keçiririk
loaded_tokenizer = Tokenizer.from_file(OUTPUT_FILE)

test_text = "Süni İntellekt Azərbaycan dilində danışır."
encoding = loaded_tokenizer.encode(test_text)

print(f"\nSınaq mətni: '{test_text}'")
print(f"Tokenlər: {encoding.tokens}")
print(f"Token ID-ləri: {encoding.ids}")
print(f"Sözlük həcmi: {loaded_tokenizer.get_vocab_size()}")
```

### 2. Kodun İzahı (Hər Sətrin Detallı İzahı)

| Sətr | Kod | İzah |
| :--- | :--- | :--- |
| 2 | `from tokenizers import Tokenizer` | Əsas `Tokenizer` sinfini daxil edirik. |
| 3 | `from tokenizers.models import BPE` | Tokenizasiya üçün **BPE (Byte Pair Encoding)** modelini daxil edirik. |
| 4 | `from tokenizers.trainers import BpeTrainer` | BPE modelini təlim etmək üçün lazım olan `BpeTrainer` sinfini daxil edirik. |
| 5 | `from tokenizers.pre_tokenizers import Whitespace` | Mətni boşluqlara görə ilkin hissələrə bölən `Whitespace` funksiyasını daxil edirik. |
| 8 | `VOCAB_SIZE = 32000` | **Sözlük həcmini 32,000 olaraq təyin edirik.** Bu, 100M parametreli model üçün yaxşı bir başlanğıcdır. |
| 9 | `FILE_PATH = "azcorpus_cleaned.txt"` | Təmizlənmiş məlumatımızın yolunu göstəririk. |
| 11 | `OUTPUT_FILE = "az_bpe_tokenizer.json"` | Tokenizatorun saxlanacağı faylın adını təyin edirik. |
| 15-19 | `SPECIAL_TOKENS = [...]` | Modelin ehtiyacı olan xüsusi tokenləri siyahı şəklində təyin edirik. |
| 23 | `tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))` | Yeni bir tokenizator obyekti yaradırıq və ona **BPE** modelini istifadə etməsini, tanımadığı tokenlər üçün isə `<|unk|>` tokenini istifadə etməsini bildiririk. |
| 27 | `tokenizer.pre_tokenizer = Whitespace()` | Tokenizasiyadan əvvəl mətni boşluqlara görə ayırmasını təyin edirik. |
| 30 | `trainer = BpeTrainer(...)` | Təlimçi obyekti yaradırıq. |
| 31 | `vocab_size=VOCAB_SIZE,` | Təlimçiyə sözlük həcminin 32000 olacağını bildiririk. |
| 32 | `special_tokens=SPECIAL_TOKENS,` | Təlim zamanı xüsusi tokenləri də nəzərə almasını təmin edirik. |
| 36 | `tokenizer.train([FILE_PATH], trainer)` | **Əsas təlim əmri.** Təlimçi, `azcorpus_cleaned.txt` faylındakı mətnlər üzərində BPE alqoritmini icra edir. |
| 39 | `tokenizer.save(OUTPUT_FILE)` | Təlim olunmuş tokenizatoru gələcəkdə istifadə etmək üçün yadda saxlayırıq. |
| 43 | `loaded_tokenizer = Tokenizer.from_file(OUTPUT_FILE)` | Yadda saxladığımız tokenizatoru yükləyirik. |
| 45 | `encoding = loaded_tokenizer.encode(test_text)` | Sınaq mətnini tokenlərə çeviririk. |
| 47-49 | `print(...)` | Nəticələri ekrana çıxarırıq. |

### 3. İcra

`llm_50gun` mühitiniz aktivdirsə, kodu icra edin:

```bash
python train_tokenizer.py
```

Təlim prosesi sizin kompüterinizin sürətindən asılı olaraq bir neçə dəqiqə çəkə bilər. Nəticədə, **`az_bpe_tokenizer.json`** adlı fayl yaranacaq.

### 💡 Günün Tapşırığı: Praktika

1.  `train_tokenizer.py` faylını yaradın və icra edin.
2.  Yaranan **`az_bpe_tokenizer.json`** faylının ölçüsünü yoxlayın (çox kiçik olmalıdır).
3.  Kodu dəyişdirərək, başqa bir Azərbaycan dilində cümləni tokenizasiya edin və nəticəni təhlil edin. Görün, hansı sözlər bir token, hansılar isə bir neçə tokenə bölünüb.

**Sabah görüşənədək!** 👋 Sabah **Tokenizatoru istifadə edərək bütün məlumatı modelin təlimi üçün hazır vəziyyətə gətirəcəyik.**

***

**Söz Sayı:** 850 söz.
