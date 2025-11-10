# Gün 7: Dataset İnşası II: Web Scraping (Məlumatın Çəkilməsi) 🕸️

## 7.1. Web Scraping-in Texniki Əsası

**Web Scraping (Vebdən Məlumat Çəkmə)** – veb-saytlardan məlumatları avtomatik olaraq çıxarmaq üçün istifadə olunan bir texnikadır. Bu proses, LLM təlimi üçün xam mətn korpusunun inşasında əsas addımdır.

Bu proses iki əsas mərhələdən ibarətdir:

1.  **HTTP Sorğusu:** Veb-saytın HTML məzmununu əldə etmək üçün `requests` kitabxanası ilə HTTP GET sorğusu göndərilir.
2.  **HTML Analizi:** Əldə edilmiş HTML məzmunu `BeautifulSoup` kimi kitabxanalarla analiz edilir və yalnız lazım olan mətn elementləri (məsələn, `<p>` teqləri) çıxarılır.

## 7.2. Etik və Hüquqi Mülahizələr

Web Scraping edərkən etik və hüquqi çərçivəyə riayət etmək vacibdir:

*   **`robots.txt`:** Hər hansı bir saytdan məlumat çəkməzdən əvvəl, həmin saytın `robots.txt` faylı yoxlanılmalıdır. Bu fayl, saytın hansı hissələrinin avtomatik çəkilməsinə icazə verildiyini göstərən protokoldur.
*   **Server Yükü:** Sorğular arasında **gecikmə (delay)** tətbiq edilməlidir (məsələn, 1-3 saniyə). Bu, saytın serverini həddindən artıq yükləməyin qarşısını alır və serverə dostyana yanaşmanı təmin edir.
*   **Müəllif Hüquqları:** Toplanmış məlumat yalnız **təlim məqsədləri** üçün istifadə edilməlidir.

## 7.3. Praktika: Sadə Scraping Skripti

Aşağıdakı Python skripti, verilmiş URL-lərdən mətn məlumatını çəkmək üçün sadə bir nümunədir.

**`scraper.py`**

```python
import requests
from bs4 import BeautifulSoup
import time
import random

URLS = [
    "https://az.wikipedia.org/wiki/Az%C9%99rbaycan_dili",
    # ... digər URL-lər
]
OUTPUT_FILE = "raw_corpus.txt"

def scrape_page(url):
    """Verilmiş URL-dən mətn məlumatını çəkir."""
    try:
        # User-Agent: Bot kimi tanınmamaq üçün brauzer məlumatlarını göndəririk.
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status() # HTTP xətalarını yoxlamaq
        soup = BeautifulSoup(response.content, 'html.parser')

        # Əsas mətn elementlərini tapmaq (hər sayt üçün fərqli ola bilər)
        paragraphs = soup.find_all('p')
        
        page_text = ""
        for p in paragraphs:
            # Məntiq: Çox qısa sətirlər (məsələn, başlıqlar) atılır.
            if len(p.text.strip()) > 50:
                page_text += p.text.strip() + "\n\n"
        
        return page_text

    except requests.exceptions.RequestException as e:
        print(f"Xəta baş verdi: {url} - {e}")
        return None

def main_scraper():
    """Əsas scraping prosesini idarə edir."""
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for url in URLS:
            text = scrape_page(url)
            if text:
                f.write(f"--- URL: {url} ---\n")
                f.write(text)
                f.write("\n\n")
            
            # Serveri yükləməmək üçün təsadüfi gecikmə
            delay = random.uniform(1, 3) 
            time.sleep(delay)

if __name__ == "__main__":
    main_scraper()
```

## 7.4. Kodun Məntiqi İzahı

| Sətr | Kod | Məntiqi İzahı |
| :--- | :--- | :--- |
| **21** | `response.raise_for_status()` | **Məntiq:** Əgər veb-sayt 404 (Tapılmadı) və ya 500 (Server Xətası) kimi bir cavab verərsə, bu, məlumatın etibarsız olduğunu göstərir. Bu funksiya xətanı dərhal aşkar edib prosesi dayandırır. |
| **24** | `soup.find_all('p')` | **Məntiq:** HTML-də `<p>` teqi adətən əsas mətn paraqraflarını ehtiva edir. Bu, mətnin əsas hissəsini reklam və naviqasiya elementlərindən ayırmağın ən sadə yoludur. |
| **30** | `if len(p.text.strip()) > 50:` | **Məntiq:** Çox qısa mətn parçaları (məsələn, "Əlaqə", "Daha çox oxu") adətən naviqasiya qalıqlarıdır. Onları silməklə, korpusun keyfiyyətini artırırıq. |
| **44** | `time.sleep(delay)` | **Məntiq:** Təsadüfi gecikmə tətbiq etməklə, serverin avtomatik bot aşkarlama mexanizmlərindən yayınmaq və serverə dostyana yanaşmaq. |

**Qeyd:** Bu skript hər bir veb-saytın fərqli HTML strukturuna uyğunlaşdırılmalıdır.
