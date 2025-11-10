# Gün 8: Dataset İnşası III: Məlumatın Təmizlənməsi (Cleaning) 🧹

## 8.1. Təmizləmənin Zəruriliyi

Web Scraping nəticəsində əldə edilən xam mətn korpusu (məsələn, `raw_corpus.txt`) təlim üçün yararsızdır. Bu məlumatlar **"səs-küy" (noise)** adlanan arzuolunmaz elementlərlə doludur: HTML qalıqları, təkrarlanan mətnlər, qeyri-standart simvollar və s.

**Məntiq:** Modelin öyrənmə keyfiyyəti birbaşa məlumatın keyfiyyətindən asılıdır. Təmizlənməmiş məlumat modelin lazımsız məlumatları əzbərləməsinə və təlim prosesinin səmərəsizliyinə səbəb olur.

## 8.2. Təmizləmə Strategiyası

Təmizləmə prosesi, məlumatın təlim üçün optimal formaya gətirilməsi üçün bir neçə ardıcıl addımdan ibarətdir:

| Addım | Məqsəd | Məntiqi Əsas |
| :--- | :--- | :--- |
| **Boşluqların Normallaşdırılması** | Ardıcıl boşluq simvollarını (yeni sətir, tab, çoxlu boşluq) tək bir boşluqla əvəz etmək. | Modelin mətnin formatından deyil, məzmunundan öyrənməsini təmin etmək. |
| **Qısa Sətirlərin Silinməsi** | Məsələn, 50 simvoldan qısa olan sətirləri (naviqasiya qalıqları) çıxarmaq. | Korpusun yalnız mənalı və informativ mətnlərdən ibarət olmasını təmin etmək. |
| **Təkrarlanan Sətirlərin Silinməsi** | Eyni cümlələrin və ya paraqrafların korpusdan çıxarılması. | Modelin eyni məlumatı dəfələrlə görməsinin qarşısını almaq və təlimin effektivliyini artırmaq. |

## 8.3. Praktika: Təmizləmə Skripti

Aşağıdakı skript `raw_corpus.txt` faylını oxuyur və yuxarıdakı strategiyaya uyğun olaraq təmizləyir.

**`cleaner.py`**

```python
import re

INPUT_FILE = "raw_corpus.txt"
OUTPUT_FILE = "clean_corpus.txt"
MIN_LINE_LENGTH = 50 # Minimum simvol sayı

def clean_text(text):
    """Mətni təmizləyən əsas funksiya."""
    
    # 1. Boşluqların Normallaşdırılması (Regex istifadəsi)
    # Bir və ya daha çox boşluq simvolunu tək bir boşluqla əvəz edir.
    text = re.sub(r'\s+', ' ', text)
    
    # 2. Xüsusi simvolları təmizləmək (HTML qalıqları)
    text = re.sub(r'&[a-z]+;', '', text)
    
    # 3. Əlavə boşluqları təmizləmək
    text = text.strip()
    
    return text

def main_cleaner():
    """Əsas təmizləmə prosesini idarə edir."""
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        raw_content = f.read()
        
    raw_lines = raw_content.split('\n')
    cleaned_lines = []
    seen_lines = set() # Təkrarlanan sətirləri yoxlamaq üçün dəst
    
    for line in raw_lines:
        cleaned_line = clean_text(line)
        
        # 4. Qısa sətirlərin silinməsi
        if len(cleaned_line) < MIN_LINE_LENGTH:
            continue
            
        # 5. Təkrarlanan sətirlərin silinməsi
        if cleaned_line not in seen_lines:
            cleaned_lines.append(cleaned_line)
            seen_lines.add(cleaned_line)
            
    # Təmizlənmiş məzmunu fayla yaz
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(cleaned_lines))
        
    print(f"Təmizləmə tamamlandı. Xam sətir sayı: {len(raw_lines)}, Təmizlənmiş unikal sətir sayı: {len(cleaned_lines)}")

if __name__ == "__main__":
    main_cleaner()
```

## 8.4. Kodun Məntiqi İzahı

| Sətr | Kod | Məntiqi İzahı |
| :--- | :--- | :--- |
| **16** | `text = re.sub(r'\s+', ' ', text)` | **Regular Expression (Regex)** istifadə olunur. `\s+` bir və ya daha çox boşluq simvolunu ifadə edir. Onu tək bir boşluqla əvəz etməklə, mətnin daxili formatını standartlaşdırırıq. |
| **35** | `seen_lines = set()` | **Set** məlumat strukturu unikal elementləri saxlamaq üçün optimallaşdırılmışdır. Bu, hər bir sətir üçün bütün əvvəlki sətirləri yoxlamaqdan (O(N^2) mürəkkəbliyi) daha sürətli (O(1) mürəkkəbliyi) yoxlamağa imkan verir. |
| **40** | `if len(cleaned_line) < MIN_LINE_LENGTH:` | Bu, məlumatın keyfiyyətini artırmaq üçün sadə bir **filtrasiya** üsuludur. Qısa sətirlər modelin öyrənməsinə az töhfə verir. |
| **43** | `if cleaned_line not in seen_lines:` | Təkrarlanan məlumatların modelin çəkilərini lazımsız yerə eyni istiqamətdə çəkməsinin qarşısını alır. |
