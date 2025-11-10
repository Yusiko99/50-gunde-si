# Gün 9: Dataset İnşası IV: Məlumatın Normallaşdırılması 📐

## 9.1. Normallaşdırmanın Məqsədi

Məlumatın təmizlənməsindən sonra, **Normallaşdırma** prosesi məlumatın təlim üçün ən uyğun və standart formata gətirilməsini təmin edir. Bu, modelin eyni mənanı daşıyan, lakin fərqli yazılışlara malik simvolları eyni şəkildə qəbul etməsinə kömək edir.

Azərbaycan dili üçün normallaşdırma xüsusilə vacibdir, çünki:

1.  **Əlifba Fərqləri:** Kiril və Latın əlifbalarının qarışığı və ya qeyri-standart simvollar mövcud ola bilər.
2.  **Unicode Fərqləri:** Eyni hərfin fərqli Unicode kodlaşdırmaları ola bilər.

## 9.2. Normallaşdırma Strategiyası

| Addım | Məqsəd | Məntiqi Əsas |
| :--- | :--- | :--- |
| **Kiçik Hərflərə Çevirmə** | Bütün hərfləri kiçik hərflərə çevirmək. | Modelin eyni sözün böyük və kiçik hərflərlə yazılmış formalarını eyni token kimi qəbul etməsini təmin etmək. |
| **Kiril-Latın Çevrilməsi** | Kiril əlifbasındakı simvolları Latın əlifbasındakı ekvivalentləri ilə əvəz etmək. | Modelin əsasən Latın əlifbası üzərində təlim keçməsini təmin etmək. |
| **Simvol Filtrasiyası** | Azərbaycan dilinin əsas hərfləri və durğu işarələrindən başqa bütün xüsusi simvolları (emoji, qeyri-standart simvollar) silmək. | Məlumatın səs-küyünü azaltmaq və modelin yalnız dilin əsas elementlərinə fokuslanmasını təmin etmək. |

## 9.3. Praktika: Normallaşdırma Skripti

**`normalizer.py`**

```python
import re
import unicodedata

INPUT_FILE = "clean_corpus.txt"
OUTPUT_FILE = "normalized_corpus.txt"

# Kiril-Latın çevrilməsi üçün sadə lüğət (yalnız nümunə üçündür)
KIRIL_TO_LATIN = {
    'а': 'a', 'б': 'b', 'в': 'v', 'г': 'q', 'д': 'd', 'е': 'e', 'ж': 'j', 'з': 'z', 
    'и': 'i', 'й': 'y', 'к': 'k', 'л': 'l', 'м': 'm', 'н': 'n', 'о': 'o', 'п': 'p', 
    'р': 'r', 'с': 's', 'т': 't', 'у': 'u', 'ф': 'f', 'х': 'x', 'ц': 'ts', 'ч': 'ç', 
    'ш': 'ş', 'ы': 'ı', 'ь': '', 'э': 'e', 'ю': 'yu', 'я': 'ya',
}

def normalize_text(text):
    """Mətni normallaşdıran əsas funksiya."""
    
    # 1. Kiçik Hərflərə Çevirmə
    text = text.lower()
    
    # 2. Kiril-Latın Çevrilməsi
    for kiril, latin in KIRIL_TO_LATIN.items():
        text = text.replace(kiril, latin)
        
    # 3. Simvol Filtrasiyası (Azərbaycan hərfləri, rəqəmlər və əsas durğu işarələri)
    # [^a-zəöğüşıç0-9\s\.\,\!\?\-] - Bu siyahıda olmayan hər şeyi boşluqla əvəz edir.
    text = re.sub(r'[^a-zəöğüşıç0-9\s\.\,\!\?\-]', ' ', text)
    
    # 4. Ardıcıl boşluqları tək boşluqla əvəz etmək
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 5. Unicode Normallaşdırması
    text = unicodedata.normalize('NFKC', text)
    
    return text

def main_normalizer():
    """Əsas normallaşdırma prosesini idarə edir."""
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        raw_lines = f.readlines()
        
    normalized_lines = [normalize_text(line) for line in raw_lines if normalize_text(line)]
            
    # Normallaşdırılmış məzmunu fayla yaz
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(normalized_lines))
        
    print(f"Normallaşdırma tamamlandı. Yekun sətir sayı: {len(normalized_lines)}")

if __name__ == "__main__":
    main_normalizer()
```

## 9.4. Kodun Məntiqi İzahı

| Sətr | Kod | Məntiqi İzahı |
| :--- | :--- | :--- |
| **36** | `text = text.lower()` | **Məntiq:** Modelin "Kitab" və "kitab" sözlərini fərqli tokenlər kimi qəbul etməsinin qarşısını alır. Bu, lüğətin ölçüsünü azaldır və modelin öyrənməsini sürətləndirir. |
| **43** | `re.sub(r'[^a-zəöğüşıç0-9\s\.\,\!\?\-]', ' ', text)` | **Məntiq:** Bu, **whitelist** (ağ siyahı) yanaşmasıdır. Yalnız Azərbaycan dilinin Latın əlifbasındakı hərflərini və əsas durğu işarələrini saxlayır. Bu siyahıda olmayan hər hansı bir simvol (məsələn, emoji, xüsusi simvollar) modelin təliminə səs-küy qatdığı üçün silinir. |
| **49** | `unicodedata.normalize('NFKC', text)` | **Məntiq:** Unicode-da bəzi simvolların bir neçə fərqli təsviri ola bilər. `NFKC` (Normalization Form KC) bu simvolları vahid, standart bir formaya gətirir. |
