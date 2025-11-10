# Gün 8: Dataset İnşası III: Məlumatın Təmizlənməsi (Cleaning) 🧹

## 8.1. Niyə Təmizləməyə Ehtiyac Var?

Dünənki Web Scraping prosesi nəticəsində əldə etdiyimiz `raw_corpus.txt` faylı **"çirkli"** məlumatlarla doludur. Bu "çirk" aşağıdakıları əhatə edir:

1.  **Artıq Simvollar:** HTML teqlərinin qalıqları, `\n` (yeni sətir), `\t` (tab) kimi boşluq simvolları.
2.  **Təkrarlanan Mətn:** Saytın naviqasiya menyuları, reklamlar, footer mətnləri.
3.  **Qeyri-Azərbaycan Dili:** Bəzi səhifələrdə qarışıq ingilis və ya rus dili mətnləri.

Əgər modelimizi bu "çirkli" məlumatlarla təlim etsək, o, yalnız pis nəticələr verməyəcək, həm də **təlim prosesi daha uzun və daha az effektiv** olacaq.

## 8.2. Təmizləmə Addımları

Biz təmizləmə prosesini bir neçə mərhələyə böləcəyik:

| Addım | Məqsəd | İstifadə Olunan Texnika |
| :--- | :--- | :--- |
| **1. Boşluqların Normallaşdırılması** | Bütün boşluq simvollarını (tab, yeni sətir) tək bir boşluqla əvəz etmək. | Python-un `re` (Regex) kitabxanası. |
| **2. Kiçik Hərflərə Çevirmə (Lowercasing)** | Bütün mətnin kiçik hərflərə çevrilməsi. | Python-un `lower()` metodu. |
| **3. Təkrarlanan Sətirlərin Silinməsi** | Eyni cümlələrin və ya paraqrafların korpusdan çıxarılması. | Python `set` strukturu. |
| **4. Qısa Sətirlərin Silinməsi** | Çox qısa və mənasız sətirləri (məsələn, "Əlaqə", "Daxil ol") silmək. | Sətrin simvol sayına görə filtrasiya. |

## 8.3. Praktika: Təmizləmə Skripti

Gəlin, `raw_corpus.txt` faylını təmizləyən bir Python skripti yazaq.

**`cleaner.py`**

```python
import re

INPUT_FILE = "raw_corpus.txt"
OUTPUT_FILE = "clean_corpus.txt"

def clean_text(text):
    """Mətni təmizləyən əsas funksiya."""
    
    # 1. Boşluqların Normallaşdırılması
    # Bütün ardıcıl boşluq simvollarını (tab, yeni sətir, boşluq) tək bir boşluqla əvəz et
    text = re.sub(r'\s+', ' ', text)
    
    # 2. Kiçik Hərflərə Çevirmə (Lowercasing)
    # LLM-lər üçün böyük hərflərin saxlanması vacib ola bilər, lakin 
    # kiçik modelimiz üçün sadəlik naminə kiçik hərflərə çeviririk.
    text = text.lower()
    
    # 3. Xüsusi simvolları təmizləmək (əgər varsa)
    # Məsələn, HTML-dən qalan '&amp;' kimi simvolları təmizləyirik
    text = re.sub(r'&[a-z]+;', '', text)
    
    # 4. Əlavə boşluqları təmizləmək
    text = text.strip()
    
    return text

def main_cleaner():
    """Əsas təmizləmə prosesini idarə edir."""
    
    print(f"'{INPUT_FILE}' faylı oxunur...")
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        raw_content = f.read()
        
    # Mətni sətirlərə bölürük
    raw_lines = raw_content.split('\n')
    
    cleaned_lines = []
    seen_lines = set() # Təkrarlanan sətirləri yoxlamaq üçün set
    
    for line in raw_lines:
        # Təmizləmə funksiyasını tətbiq et
        cleaned_line = clean_text(line)
        
        # 4. Qısa sətirlərin silinməsi (minimum 50 simvol)
        if len(cleaned_line) < 50:
            continue
            
        # 3. Təkrarlanan sətirlərin silinməsi
        if cleaned_line not in seen_lines:
            cleaned_lines.append(cleaned_line)
            seen_lines.add(cleaned_line)
            
    print(f"Ümumi xam sətir sayı: {len(raw_lines)}")
    print(f"Təmizlənmiş unikal sətir sayı: {len(cleaned_lines)}")
    
    # Təmizlənmiş məzmunu fayla yaz
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(cleaned_lines))
        
    print(f"Təmizləmə tamamlandı. Nəticə '{OUTPUT_FILE}' faylına yazıldı.")

if __name__ == "__main__":
    main_cleaner()
```

## 8.4. Kodun İzahı

| Sətr | Kod | İzahı |
| :--- | :--- | :--- |
| **14** | `text = re.sub(r'\s+', ' ', text)` | **Regex (Regular Expression)** istifadə edərək bir və ya daha çox boşluq simvolunu (`\s+`) tək bir boşluqla əvəz edir. Bu, mətnin formatını normallaşdırır. |
| **19** | `text = text.lower()` | Bütün hərfləri kiçik hərflərə çevirir. Bu, modelin eyni sözün böyük və kiçik hərflərlə yazılmış formalarını eyni şəkildə qəbul etməsinə kömək edir. |
| **34** | `seen_lines = set()` | **Set** (dəst) Python-da unikal elementləri saxlamaq üçün istifadə olunan bir məlumat strukturudur. Bu, təkrarlanan sətirləri sürətlə yoxlamağa imkan verir. |
| **43** | `if len(cleaned_line) < 50:` | Sətrin uzunluğunu yoxlayır. 50 simvoldan qısa sətirlər adətən mənasız başlıqlar və ya qalıqlar olur, ona görə də onları atırıq. |
| **46** | `if cleaned_line not in seen_lines:` | Əgər təmizlənmiş sətir artıq `seen_lines` dəstində yoxdursa, onu korpusa əlavə edirik. |

**Gündəlik Tapşırıq:** `cleaner.py` skriptini yaradın və işə salın. `clean_corpus.txt` faylının ölçüsünü və məzmununu yoxlayın. Görəcəksiniz ki, məlumat daha səliqəli və təlim üçün daha uyğun hala gəlib.
