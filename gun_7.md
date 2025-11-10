# Gün 7: Dataset İnşası II: Web Scraping (Məlumatın Çəkilməsi) 🕸️

## 7.1. Web Scraping Nədir?

**Web Scraping (Vebdən Məlumat Çəkmə)** – veb-saytlardan avtomatik olaraq məlumat toplama prosesidir. Bizim məqsədimiz, Gün 6-da müəyyənləşdirdiyimiz URL-lərdən mətn məlumatlarını çəkməkdir.

Bu proses üçün iki əsas Python kitabxanasından istifadə edəcəyik:

1.  **`requests`:** Veb-saytın HTML məzmununu əldə etmək üçün.
2.  **`BeautifulSoup`:** HTML məzmununu analiz etmək və yalnız lazım olan mətn hissələrini (məsələn, məqalənin mətni) çıxarmaq üçün.

## 7.2. Etik və Hüquqi Mülahizələr

Web Scraping edərkən **etik və hüquqi məsuliyyətlərinizi** unutmayın:

*   **`robots.txt`:** Hər hansı bir saytı çəkməzdən əvvəl, həmin saytın `robots.txt` faylını yoxlayın. Bu fayl, saytın hansı hissələrinin çəkilməsinə icazə verildiyini göstərir.
*   **Server Yükü:** Sorğuları çox sürətli göndərməyin. Bu, saytın serverini yükləyə bilər. Sorğular arasında kiçik bir gecikmə (məsələn, 1 saniyə) qoymaq məsləhətdir.
*   **Müəllif Hüquqları:** Topladığınız məlumatı yalnız **təlim məqsədləri** üçün istifadə edin və heç bir halda kommersiya məqsədləri üçün yenidən yayımlamayın.

## 7.3. Praktika: Sadə Scraping Skripti

Gəlin, sadə bir veb-saytdan məlumat çəkən Python skripti yazaq.

**`scraper.py`**

```python
import requests
from bs4 import BeautifulSoup
import time
import random

# 1. Mənbə URL-ləri
# Bu siyahını Gün 6-da hazırladığınız URL-lərlə əvəz edin.
URLS = [
    "https://az.wikipedia.org/wiki/Az%C9%99rbaycan_dili",
    "https://report.az/siyaset/", # Nümunə olaraq
    # ... digər URL-lər
]

# 2. Məlumatı saxlayacağımız fayl
OUTPUT_FILE = "raw_corpus.txt"

def scrape_page(url):
    """Verilmiş URL-dən mətn məlumatını çəkir."""
    try:
        # 3. Veb-sayta sorğu göndərmək
        # Bəzi saytlar botları bloklayır, buna görə də User-Agent əlavə edirik.
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status() # Xəta olarsa, xəbərdarlıq et

        # 4. HTML-i analiz etmək
        soup = BeautifulSoup(response.content, 'html.parser')

        # 5. Əsas mətn hissələrini tapmaq
        # Bu hissə hər sayt üçün fərqli olacaq.
        # Nümunə: <p> teqlərinin içindəki mətn
        paragraphs = soup.find_all('p')
        
        page_text = ""
        for p in paragraphs:
            # Mətnin çox qısa olub-olmadığını yoxlayırıq
            if len(p.text.strip()) > 50:
                page_text += p.text.strip() + "\n\n"
        
        return page_text

    except requests.exceptions.RequestException as e:
        print(f"Xəta baş verdi: {url} - {e}")
        return None

def main_scraper():
    """Əsas scraping prosesini idarə edir."""
    print(f"Scraping prosesi başladı. Məlumatlar '{OUTPUT_FILE}' faylına yazılacaq.")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for url in URLS:
            print(f"-> {url} çəkilir...")
            text = scrape_page(url)
            
            if text:
                f.write(f"--- URL: {url} ---\n")
                f.write(text)
                f.write("\n\n")
                print(f"   [Uğurlu] {len(text.split())} söz yazıldı.")
            else:
                print(f"   [Uğursuz] Məlumat çəkilmədi.")
            
            # 7. Serveri yükləməmək üçün gecikmə
            delay = random.uniform(1, 3) # 1 ilə 3 saniyə arasında təsadüfi gecikmə
            time.sleep(delay)

    print("Scraping prosesi tamamlandı.")

if __name__ == "__main__":
    main_scraper()
```

## 7.4. Kodun İzahı

| Sətr | Kod | İzahı |
| :--- | :--- | :--- |
| **3** | `import requests` | Veb-saytlara HTTP sorğuları göndərmək üçün kitabxana. |
| **4** | `from bs4 import BeautifulSoup` | HTML-i analiz etmək və məlumat çıxarmaq üçün kitabxana. |
| **5-6** | `import time, random` | Serveri yükləməmək üçün gecikmə yaratmaq üçün. |
| **14** | `def scrape_page(url):` | Hər bir URL üçün məlumat çəkmə funksiyası. |
| **19** | `headers = {...}` | Saytın bizi bot kimi qəbul etməməsi üçün brauzer məlumatlarını göndəririk. |
| **22** | `response = requests.get(...)` | URL-ə GET sorğusu göndəririk. |
| **23** | `response.raise_for_status()` | Sorğu uğursuz olarsa (məsələn, 404 xətası), proqramı dayandırır. |
| **26** | `soup = BeautifulSoup(...)` | HTML məzmununu `BeautifulSoup` obyektinə çevirir. |
| **30** | `paragraphs = soup.find_all('p')` | Səhifədəki bütün `<p>` (paraqraf) teqlərini tapır. **Qeyd:** Bu, hər sayt üçün dəyişməlidir! |
| **34** | `if len(p.text.strip()) > 50:` | Çox qısa paraqrafları (məsələn, başlıqları) atmaq üçün sadə təmizləmə. |
| **48** | `time.sleep(delay)` | Təsadüfi gecikmə tətbiq edərək serverə dostyana yanaşırıq. |

**Gündəlik Tapşırıq:** `scraper.py` faylını yaradın və `URLS` siyahısını Gün 6-da təyin etdiyiniz ən azı 3-5 Azərbaycan saytı ilə əvəz edin. Skripti işə salın və `raw_corpus.txt` faylının yarandığını yoxlayın.
