# Gün 11: Tokenizasiya II: Tokenizatorun Qurulması (Praktika) 🛠️

## 11.1. Tokenizatorun Təlimi

Dünən BPE (Byte Pair Encoding) nəzəriyyəsini öyrəndik. Bu gün isə **`tokenizers`** kitabxanasından istifadə edərək, **`normalized_corpus.txt`** faylındakı məlumatlar əsasında öz Azərbaycan dili tokenizatorumuzu təlim edəcəyik.

Bu tokenizator bizim LLM-in dilini başa düşməsi üçün əsas vasitə olacaq.

**`train_tokenizer.py`**

```python
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

# 1. Giriş və Çıxış Faylları
CORPUS_FILE = "normalized_corpus.txt"
VOCAB_SIZE = 32000 # Lüğətin hədəf ölçüsü
OUTPUT_PREFIX = "az_llm"

def train_bpe_tokenizer():
    """BPE tokenizatorunu təlim edir."""
    
    # 2. Tokenizatorun Modeli: BPE
    tokenizer = Tokenizer(models.BPE())
    
    # 3. Mətnin ilkin emalı (Pre-tokenizer)
    # Mətni sözlərə bölmək üçün sadə boşluq əsaslı pre-tokenizer istifadə edirik.
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    
    # 4. Təlimçi (Trainer)
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
        # Əsasən ingilis dilində istifadə olunur, lakin biz də əlavə edirik.
        # [UNK] - Naməlum token, [PAD] - Doldurma tokeni
        min_frequency=2 # Ən azı 2 dəfə rast gəlinən cütlükləri lüğətə əlavə et
    )
    
    # 5. Təlim prosesi
    print(f"Tokenizator '{CORPUS_FILE}' üzərində təlim edilir...")
    tokenizer.train([CORPUS_FILE], trainer=trainer)
    print("Təlim tamamlandı.")
    
    # 6. Tokenizatoru yadda saxlamaq
    tokenizer.save(f"{OUTPUT_PREFIX}-tokenizer.json")
    print(f"Tokenizator '{OUTPUT_PREFIX}-tokenizer.json' faylına yazıldı.")
    
    # 7. Nümunə sınaq
    test_sentence = "süni intellekt modelinin kvantlaşdırılması prosesi uğurla başa çatdı"
    encoding = tokenizer.encode(test_sentence)
    
    print("\n--- Nümunə Sınaq ---")
    print(f"Orijinal: {test_sentence}")
    print(f"Tokenlər: {encoding.tokens}")
    print(f"ID-lər: {encoding.ids}")
    print(f"Lüğət Ölçüsü: {tokenizer.get_vocab_size()}")

if __name__ == "__main__":
    train_bpe_tokenizer()
```

## 11.2. Kodun İzahı

| Sətr | Kod | İzahı |
| :--- | :--- | :--- |
| **10** | `tokenizer = Tokenizer(models.BPE())` | Yeni bir BPE (Byte Pair Encoding) modeli yaradırıq. |
| **14** | `tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()` | **Pre-tokenizer** tokenizasiyadan əvvəl mətni ilkin olaraq bölür. `Whitespace` (Boşluq) əsasında bölmə, sözləri boşluqlara görə ayırır. |
| **17** | `trainer = trainers.BpeTrainer(...)` | BPE təlimçisini yaradırıq. |
| **18** | `vocab_size=VOCAB_SIZE` | Lüğətin maksimum ölçüsünü 32000 olaraq təyin edirik. |
| **19** | `special_tokens=["[UNK]", ...]` | Modelin xüsusi məqsədlər üçün istifadə edəcəyi tokenlər. Məsələn, **`[UNK]`** (Unknown) lüğətdə olmayan sözlər üçün istifadə olunacaq. |
| **26** | `tokenizer.train([CORPUS_FILE], trainer=trainer)` | Tokenizatoru `normalized_corpus.txt` faylı üzərində təlim edir. |
| **30** | `tokenizer.save(...)` | Təlim edilmiş tokenizatoru JSON formatında yadda saxlayır. Bu fayl modelimizlə birlikdə istifadə olunacaq. |
| **34** | `encoding = tokenizer.encode(test_sentence)` | Tokenizatorun necə işlədiyini yoxlamaq üçün nümunə cümləni rəqəmlərə çevirir. |

**Gündəlik Tapşırıq:** `train_tokenizer.py` skriptini yaradın və işə salın. Nəticədə **`az_llm-tokenizer.json`** faylı yaranmalıdır. Nümunə sınağın nəticələrini diqqətlə yoxlayın. Azərbaycan dilindəki uzun sözlərin (məsələn, "kvantlaşdırılması") necə hissələrə bölündüyünü müşahidə edin.
