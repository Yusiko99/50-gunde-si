# Gün 35: Ollama API ilə İşləmək (Chatbotun Qurulması) 💬

## 35.1. Ollama API-nin Funksional Əhəmiyyəti

Ollama, modelin yerli kompüterdə işləməsinə baxmayaraq, ona **HTTP API** vasitəsilə müraciət etməyə imkan verir.

**Məntiq:** Bu API, modelin birbaşa terminalda deyil, Python, JavaScript və ya hər hansı digər proqramlaşdırma dili ilə yazılmış xarici bir tətbiqə (məsələn, veb-chatbot, mobil tətbiq) inteqrasiya edilməsinə imkan verir.

## 35.2. Praktika: Python ilə API Sorğusu

Biz Python-un **`requests`** kitabxanasından istifadə edərək Ollama API-yə sorğu göndərən sadə bir funksiya yazacağıq.

**`ollama_api_client.py`**

```python
import requests
import json

# Ollama API-nin standart ünvanı
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "az-llm-100m"

def generate_response(prompt):
    """Ollama API-yə sorğu göndərir və cavabı qaytarır."""
    
    # 1. Sorğu üçün JSON məlumatını hazırlamaq
    data = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False, # Cavabı axın şəklində deyil, tam şəkildə almaq
        "options": {
            "temperature": 0.8,
            "num_predict": 100 # Maksimum 100 token yaratmaq
        }
    }
    
    # 2. API-yə POST sorğusu göndərmək
    try:
        response = requests.post(OLLAMA_URL, json=data)
        response.raise_for_status() # HTTP xətalarını yoxlamaq
        
        # 3. Cavabı emal etmək
        result = response.json()
        
        # Yalnız yaradılmış mətni qaytarmaq
        return result.get("response", "Cavab alınmadı.")
        
    except requests.exceptions.RequestException as e:
        return f"Xəta: Ollama API-yə qoşulmaq mümkün olmadı. Ollama serveri işləyirmi? ({e})"

if __name__ == "__main__":
    test_prompt = "Azərbaycan dilində süni intellektin əhəmiyyəti nədir?"
    print(f"Sorğu: {test_prompt}")
    response = generate_response(test_prompt)
    print(f"Cavab: {response}")
```

## 35.3. Kodun Məntiqi İzahı

| Sətr | Kod | Məntiqi İzahı |
| :--- | :--- | :--- |
| **10** | `OLLAMA_URL = "http://localhost:11434/api/generate"` | **Məntiq:** Ollama serveri yerli kompüterdə (localhost) standart olaraq 11434 portunda işləyir. `/api/generate` isə mətn generasiyası üçün standart API son nöqtəsidir. |
| **18** | `"stream": False` | **Məntiq:** `stream=True` olsaydı, cavab token-token gələrdi (canlı chatbot üçün faydalıdır). `False` isə bütün cavabın bir dəfəyə gəlməsini təmin edir. |
| **20** | `"temperature": 0.8` | **Məntiq:** **Sampling** prosesində (Gün 20) token seçimi zamanı ehtimalların paylanmasını yumşaldır. Yüksək temperatur (məsələn, 1.0) daha yaradıcı, aşağı temperatur (məsələn, 0.2) isə daha deterministik cavablar verir. |
| **27** | `response.raise_for_status()` | **Məntiq:** API sorğusunun uğurlu (HTTP 200) olub-olmadığını yoxlayır. Əgər server xətası varsa, prosesi dayandırır. |

**Nəticə:** Bu API interfeysi modelin təlimdən sonra real tətbiqlərə inteqrasiyasının əsasını təşkil edir.
