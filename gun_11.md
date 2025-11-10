# Gün 11: Tokenizasiya II: Tokenizatorun Qurulması (Praktika) 🛠️

## 11.1. Tokenizatorun Təlimi

Gün 10-da öyrəndiyimiz BPE alqoritmini indi təmizlənmiş korpusumuz (`normalized_corpus.txt`) üzərində tətbiq edəcəyik.

**Məntiq:** Tokenizatorun təlimi, modelin təlimindən fərqli olaraq, modelin çəkilərini deyil, dilin özünün statistik xüsusiyyətlərini (ən çox rast gəlinən alt-söz birləşmələri) öyrənir.

## 11.2. Praktika: BPE Tokenizatorunun Təlimi

**`train_tokenizer.py`**

```python
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
import os

CORPUS_FILE = "normalized_corpus.txt"
VOCAB_SIZE = 32000
OUTPUT_FILE = "az_llm-tokenizer.json"

def train_bpe_tokenizer():
    """BPE tokenizatorunu təlim edir və yadda saxlayır."""
    
    # 1. Tokenizatorun Modelini Təyin Etmək
    # BPE modelini boş bir lüğətlə yaradırıq.
    tokenizer = Tokenizer(models.BPE())
    
    # 2. Pre-Tokenizatoru Təyin Etmək
    # Mətni ilkin olaraq sözlərə bölmək üçün istifadə olunur.
    # Whitespace: Boşluq simvolları ilə bölmə.
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    
    # 3. Təlimçini Təyin Etmək
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
        # Əlavə olaraq, GPT-3-də istifadə olunan <|endoftext|> tokenini də əlavə edirik.
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet() # Bütün ASCII simvollarını ilkin əlifbaya daxil etmək
    )
    
    # 4. Təlimi Başlatmaq
    # Tokenizatoru korpus faylı üzərində təlim edirik.
    tokenizer.train([CORPUS_FILE], trainer=trainer)
    
    # 5. Tokenizatoru Yadda Saxlamaq
    tokenizer.save(OUTPUT_FILE)
    
    print(f"Tokenizator uğurla təlim edildi və '{OUTPUT_FILE}' faylına yazıldı.")
    print(f"Yekun lüğət ölçüsü: {tokenizer.get_vocab_size()}")

if __name__ == "__main__":
    if not os.path.exists(CORPUS_FILE):
        print(f"Xəta: Korpus faylı '{CORPUS_FILE}' tapılmadı. Zəhmət olmasa Gün 9-un tapşırıqlarını tamamlayın.")
    else:
        train_bpe_tokenizer()
```

## 11.3. Kodun Məntiqi İzahı

| Sətr | Kod | Məntiqi İzahı |
| :--- | :--- | :--- |
| **14** | `tokenizer = Tokenizer(models.BPE())` | **BPE (Byte Pair Encoding)** alqoritminin əsasını təyin edir. |
| **18** | `tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()` | **Pre-tokenizasiya** – BPE-dən əvvəl mətnin necə ilkin bölünəcəyini müəyyənləşdirir. Boşluqlarla bölmə ən sadə və effektiv üsuldur. |
| **21** | `vocab_size=VOCAB_SIZE` | **Kritik parametr.** BPE alqoritmi bu ölçüyə çatana qədər birləşdirmə əməliyyatlarını davam etdirəcək. |
| **22** | `special_tokens=["[UNK]", ...]` | **Xüsusi Tokenlər** – Modelin təlimi və işləməsi üçün vacib olan tokenlər. Məsələn, `[UNK]` (Naməlum) lüğətdə olmayan sözləri əvəz edir. |
| **24** | `initial_alphabet=pre_tokenizers.ByteLevel.alphabet()` | **Məntiq:** Tokenizatorun bütün mümkün simvolları (hətta nadir simvolları) tanımasını təmin edir. |
| **28** | `tokenizer.train([CORPUS_FILE], trainer=trainer)` | Tokenizatoru korpus üzərində təlim edir. Bu proses ən çox təkrarlanan alt-söz cütlərini tapır və lüğəti qurur. |

## 11.4. Tokenizatorun Test Edilməsi

Təlimdən sonra tokenizatorun düzgün işlədiyini yoxlamaq vacibdir.

```python
# Tokenizatoru yükləmək
tokenizer = Tokenizer.from_file("az_llm-tokenizer.json")

# Nümunə mətn
text = "Süni intellekt Azərbaycan dilində mətn yarada bilər."

# Mətni tokenizasiya etmək
encoding = tokenizer.encode(text)

# Nəticəni yoxlamaq
print(f"Tokenlər: {encoding.tokens}")
print(f"ID-lər: {encoding.ids}")
```

**Gözlənilən Nəticə:** Mürəkkəb Azərbaycan sözləri (məsələn, "intellekt", "Azərbaycan") bir neçə alt-sözə bölünməlidir. Məsələn, "Azərbaycan" -> \["Azər", "bay", "can"] kimi.
