# Gün 9: Dataset İnşası IV: Məlumatın Normallaşdırılması 📐

## 9.1. Normallaşdırma Nədir?

Dünən məlumatımızı təmizlədik. Bu gün isə onu **normallaşdıracağıq**. **Normallaşdırma** məlumatın təlim üçün ən uyğun formata gətirilməsi deməkdir. Bu, modelin öyrənmə prosesini asanlaşdırır və keyfiyyətini artırır.

Azərbaycan dili üçün normallaşdırma xüsusilə vacibdir, çünki:

1.  **Kiril/Latın Problemi:** Bəzi mənbələrdə mətnlər Kiril əlifbasında ola bilər. Bizim modelimiz Latın əlifbasına əsaslanacaq.
2.  **Durğu İşarələri:** Artıq və ya səhv durğu işarələri modelin diqqətini yayındıra bilər.
3.  **Xüsusi Simvollar:** Emoji, xüsusi simvollar və ya qeyri-standart simvolların təmizlənməsi.

## 9.2. Praktika: Normallaşdırma Skripti

Bizim təmizləmə skriptimizdə (Gün 8) bəzi normallaşdırma addımları artıq var idi (məsələn, kiçik hərflərə çevirmə). İndi ona daha spesifik Azərbaycan dili normallaşdırması əlavə edəcəyik.

**`normalizer.py`**

```python
import re
import unicodedata

INPUT_FILE = "clean_corpus.txt"
OUTPUT_FILE = "normalized_corpus.txt"

# Kiril-Latın çevrilməsi üçün sadə lüğət (tam deyil, nümunə üçündür)
# Bizim məqsədimiz əsasən Latın əlifbası ilə işləməkdir.
KIRIL_TO_LATIN = {
    'а': 'a', 'б': 'b', 'в': 'v', 'г': 'q', 'ғ': 'ğ', 'д': 'd', 'е': 'e', 'ё': 'yo',
    'ж': 'j', 'з': 'z', 'и': 'i', 'й': 'y', 'к': 'k', 'қ': 'q', 'л': 'l', 'м': 'm',
    'н': 'n', 'о': 'o', 'ө': 'ö', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u',
    'ү': 'ü', 'ф': 'f', 'х': 'x', 'һ': 'h', 'ц': 'ts', 'ч': 'ç', 'ш': 'ş', 'щ': 'şç',
    'ъ': '', 'ы': 'ı', 'ь': '', 'э': 'e', 'ю': 'yu', 'я': 'ya',
    # Böyük hərflər
    'А': 'A', 'Б': 'B', 'В': 'V', 'Г': 'Q', 'Ғ': 'Ğ', 'Д': 'D', 'Е': 'E', 'Ё': 'Yo',
    'Ж': 'J', 'З': 'Z', 'И': 'İ', 'Й': 'Y', 'К': 'K', 'Қ': 'Q', 'Л': 'L', 'М': 'M',
    'Н': 'N', 'О': 'O', 'Ө': 'Ö', 'П': 'P', 'Р': 'R', 'С': 'S', 'Т': 'T', 'У': 'U',
    'Ү': 'Ü', 'Ф': 'F', 'Х': 'X', 'Һ': 'H', 'Ц': 'Ts', 'Ч': 'Ç', 'Ш': 'Ş', 'Щ': 'Şç',
    'Ъ': '', 'Ы': 'I', 'Ь': '', 'Э': 'E', 'Ю': 'Yu', 'Я': 'Ya',
}

def normalize_text(text):
    """Mətni normallaşdıran əsas funksiya."""
    
    # 1. Kiril-Latın Çevrilməsi (Əgər mətn Kiril simvolları ehtiva edirsə)
    # Bizim scraping etdiyimiz mənbələr əsasən Latın əlifbasındadır, lakin ehtiyat üçün.
    for kiril, latin in KIRIL_TO_LATIN.items():
        text = text.replace(kiril, latin)
        
    # 2. Durğu İşarələrinin Təmizlənməsi
    # Yalnız hərfləri, rəqəmləri və əsas durğu işarələrini saxlayırıq.
    # Digər xüsusi simvolları (emoji, qeyri-adi simvollar) boşluqla əvəz edirik.
    text = re.sub(r'[^a-zəöğüşıç0-9\s\.\,\!\?\-]', ' ', text)
    
    # 3. Ardıcıl boşluqları tək boşluqla əvəz etmək
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 4. Unicode Normallaşdırması (Məsələn, bəzi simvolların fərqli kodlaşdırılması)
    text = unicodedata.normalize('NFKC', text)
    
    return text

def main_normalizer():
    """Əsas normallaşdırma prosesini idarə edir."""
    
    print(f"'{INPUT_FILE}' faylı oxunur...")
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        raw_lines = f.readlines()
        
    normalized_lines = []
    
    for line in raw_lines:
        # Təmizləmə funksiyasını tətbiq et
        normalized_line = normalize_text(line)
        
        # Normallaşdırılmış sətirləri əlavə et
        if normalized_line:
            normalized_lines.append(normalized_line)
            
    print(f"Ümumi təmizlənmiş sətir sayı: {len(raw_lines)}")
    print(f"Yekun normallaşdırılmış sətir sayı: {len(normalized_lines)}")
    
    # Normallaşdırılmış məzmunu fayla yaz
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(normalized_lines))
        
    print(f"Normallaşdırma tamamlandı. Nəticə '{OUTPUT_FILE}' faylına yazıldı.")

if __name__ == "__main__":
    main_normalizer()
```

## 9.3. Kodun İzahı

| Sətr | Kod | İzahı |
| :--- | :--- | :--- |
| **2** | `import unicodedata` | Unicode simvollarını normallaşdırmaq üçün kitabxana. |
| **10-25** | `KIRIL_TO_LATIN = {...}` | Kiril əlifbasından Latın əlifbasına çevirmə üçün sadə lüğət. Bu, bəzi mənbələrdə qarışıq mətnlərin qarşısını almaq üçün ehtiyat tədbiridir. |
| **34** | `for kiril, latin in KIRIL_TO_LATIN.items():` | Mətndəki Kiril simvollarını Latın simvolları ilə əvəz edir. |
| **39** | `re.sub(r'[^a-zəöğüşıç0-9\s\.\,\!\?\-]', ' ', text)` | **Ən vacib hissə:** Bu Regex ifadəsi Azərbaycan dilinin bütün kiçik hərflərini (`a-zəöğüşıç`), rəqəmləri (`0-9`), boşluqları (`\s`) və əsas durğu işarələrini (`\.\,\!\?\-`) saxlayır. Bu siyahıda olmayan hər şeyi boşluqla əvəz edir. |
| **45** | `unicodedata.normalize('NFKC', text)` | Unicode simvollarını standart formaya gətirir. Məsələn, bəzi simvolların fərqli kodlaşdırılması varsa, onları eyniləşdirir. |

**Gündəlik Tapşırıq:** `normalizer.py` skriptini yaradın və işə salın. `normalized_corpus.txt` faylının məzmununu yoxlayın. Artıq təlim üçün istifadə ediləcək xalis mətn korpusumuz hazırdır!
