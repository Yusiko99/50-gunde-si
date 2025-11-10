# Gün 12: Məlumatın Hazırlanması: Rəqəmləşdirmə 💾

## 12.1. Məlumatın Rəqəmləşdirilməsi

Əvvəlki günlərdə:
1.  **Korpusu topladıq** (`normalized_corpus.txt`).
2.  **Tokenizatoru təlim etdik** (`az_llm-tokenizer.json`).

İndi isə son addım: **Korpusu Token ID-lərinə çevirmək** və modelin təlimi üçün hazır vəziyyətə gətirmək.

Bizim LLM-imiz **Transformer** arxitekturasına əsaslanacaq və bu model **ardıcıl mətnləri** emal edir. Buna görə də, bütün korpusumuzu böyük bir rəqəmlər ardıcıllığına çevirəcəyik.

## 12.2. Praktika: Token ID-lərinə Çevirmə

Biz bütün `normalized_corpus.txt` faylını oxuyacaq, hər bir sətri tokenizatorumuzla rəqəmlərə çevirəcək və nəticəni **NumPy** massivi kimi yadda saxlayacağıq. NumPy massivi böyük rəqəmlər toplusunu yaddaşda daha effektiv saxlamağa imkan verir.

**`prepare_data.py`**

```python
import numpy as np
from tokenizers import Tokenizer
import os

# 1. Giriş və Çıxış Faylları
CORPUS_FILE = "normalized_corpus.txt"
TOKENIZER_FILE = "az_llm-tokenizer.json"
OUTPUT_DIR = "data"

def prepare_dataset():
    """Korpusu token ID-lərinə çevirir və NumPy massivi kimi saxlayır."""
    
    # 2. Tokenizatoru yükləmək
    try:
        tokenizer = Tokenizer.from_file(TOKENIZER_FILE)
    except Exception as e:
        print(f"Xəta: Tokenizator faylı '{TOKENIZER_FILE}' tapılmadı. Zəhmət olmasa, Gün 11-i tamamlayın.")
        return

    # 3. Korpusu oxumaq
    print(f"'{CORPUS_FILE}' faylı oxunur...")
    with open(CORPUS_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 4. Bütün mətnləri token ID-lərinə çevirmək
    all_ids = []
    print("Mətnlər token ID-lərinə çevrilir...")
    
    # Batch Encoding istifadə edərək prosesi sürətləndiririk
    encodings = tokenizer.encode_batch(lines)
    
    for encoding in encodings:
        all_ids.extend(encoding.ids)

    # 5. NumPy massivinə çevirmək
    # dtype='uint16' istifadə edirik, çünki 32000 lüğət ölçüsü üçün 16 bit kifayətdir
    # Bu, yaddaşda yerə qənaət edir.
    data = np.array(all_ids, dtype=np.uint16)
    
    print(f"Ümumi token sayı: {len(data)}")
    print(f"NumPy massivinin ölçüsü: {data.nbytes / (1024*1024):.2f} MB")

    # 6. Təlim və Validasiya Dəstlərinə Bölmək
    # 90% Təlim (Train), 10% Validasiya (Validation)
    train_ratio = 0.9
    split_index = int(train_ratio * len(data))
    
    train_data = data[:split_index]
    val_data = data[split_index:]

    # 7. Nəticələri yadda saxlamaq
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    train_path = os.path.join(OUTPUT_DIR, 'train.bin')
    val_path = os.path.join(OUTPUT_DIR, 'val.bin')
    
    train_data.tofile(train_path)
    val_data.tofile(val_path)
    
    print(f"\n--- Nəticə ---")
    print(f"Təlim dəsti ({len(train_data)} token) '{train_path}' faylına yazıldı.")
    print(f"Validasiya dəsti ({len(val_data)} token) '{val_path}' faylına yazıldı.")

if __name__ == "__main__":
    prepare_dataset()
```

## 12.3. Kodun İzahı

| Sətr | Kod | İzahı |
| :--- | :--- | :--- |
| **2** | `import numpy as np` | Riyazi əməliyyatlar və böyük massivlərlə işləmək üçün kitabxana. |
| **27** | `encodings = tokenizer.encode_batch(lines)` | Bütün sətirləri bir dəfəyə token ID-lərinə çevirir. Bu, `for` dövründə tək-tək çevirməkdən daha sürətlidir. |
| **30** | `all_ids.extend(encoding.ids)` | Hər bir sətrin token ID-lərini ümumi siyahıya əlavə edir. |
| **34** | `data = np.array(all_ids, dtype=np.uint16)` | Bütün ID-ləri **16-bitlik tam ədəd** (unsigned integer) massivinə çevirir. Bu, hər bir token ID-si üçün 2 bayt yaddaş istifadə etməyimiz deməkdir. |
| **40** | `split_index = int(train_ratio * len(data))` | Məlumatı 90% təlim və 10% validasiya olaraq bölmək üçün sərhəd nöqtəsini hesablayır. |
| **47** | `train_data.tofile(train_path)` | Təlim dəstini ikili (binary) formatda yadda saxlayır. Bu, məlumatı tez və effektiv şəkildə yükləməyə imkan verir. |

**Gündəlik Tapşırıq:** `prepare_data.py` skriptini yaradın və işə salın. `data` qovluğunun içində `train.bin` və `val.bin` fayllarının yarandığını yoxlayın. **Təbrik edirik!** Siz artıq LLM təlimi üçün lazım olan bütün məlumat hazırlığı mərhələsini sıfırdan tamamladınız.
