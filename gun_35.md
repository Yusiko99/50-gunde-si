# Gün 35: Ollama API ilə İşləmək (Chatbotun Qurulması) 💬

## 35.1. Modelin İşə Salınması

Gün 34-də modelimizi Ollama-ya idxal etdik. İndi onu işə salmağın iki yolu var:

### A. Terminalda İşə Salma (Chat)

Ən sadə yol terminalda birbaşa modelimizlə söhbət etməkdir:

```bash
ollama run az-llm-100m
```

Bu əmr modeli işə salacaq və sizə birbaşa suallar verməyə imkan verəcək.

### B. Ollama API ilə İşləmək

Əgər modelinizi bir proqrama (məsələn, Python-da chatbot interfeysinə) inteqrasiya etmək istəyirsinizsə, **Ollama API**-dən istifadə etməlisiniz. Ollama API, modelinizə HTTP sorğuları vasitəsilə müraciət etməyə imkan verən yerli bir serverdir.

## 35.2. Praktika: Python Chatbotu

Biz Python-un **`requests`** kitabxanasından istifadə edərək modelimizə sorğu göndərən sadə bir chatbot skripti yazacağıq.

**`chatbot.py`**

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
        response.raise_for_status() # Xəta olarsa, xəbərdarlıq et
        
        # 3. Cavabı emal etmək
        result = response.json()
        
        # Yalnız yaradılmış mətni qaytarmaq
        return result.get("response", "Cavab alınmadı.")
        
    except requests.exceptions.RequestException as e:
        return f"Xəta: Ollama API-yə qoşulmaq mümkün olmadı. Ollama işləyirmi? ({e})"

def main_chatbot():
    """Əsas chatbot dövrü."""
    print("--- Azərbaycan LLM Chatbotu (Ollama API) ---")
    print(f"Model: {MODEL_NAME}. Çıxmaq üçün 'çıx' yazın.")
    
    while True:
        user_input = input("Siz: ")
        if user_input.lower() == 'çıx':
            break
            
        if not user_input.strip():
            continue
            
        print("LLM: Zəhmət olmasa gözləyin...")
        response = generate_response(user_input)
        print(f"LLM: {response}")

if __name__ == "__main__":
    main_chatbot()
```

## 35.3. Kodun İzahı

| Sətr | Kod | İzahı |
| :--- | :--- | :--- |
| **10** | `OLLAMA_URL = "http://localhost:11434/api/generate"` | Ollama-nın standart olaraq işlədiyi yerli API ünvanı. |
| **18** | `"stream": False` | Mətnin hissə-hissə deyil, tam şəkildə gəlməsini təmin edir. |
| **20** | `"temperature": 0.8` | Modelin yaradıcılıq dərəcəsi. Yüksək dəyər daha yaradıcı, aşağı dəyər daha dəqiq cavab deməkdir. |
| **20** | `"num_predict": 100` | Modelin maksimum neçə token yaratacağını təyin edir. |
| **27** | `response = requests.post(OLLAMA_URL, json=data)` | Hazırlanmış JSON məlumatını API-yə göndərir. |
| **30** | `result = response.json()` | API-dən gələn JSON cavabını Python lüğətinə çevirir. |
| **33** | `return result.get("response", ...)` | Cavabdan yalnız yaradılmış mətn hissəsini çıxarır. |

**Gündəlik Tapşırıq:** `chatbot.py` skriptini yaradın. Terminalda `ollama run az-llm-100m` əmrini icra edin və sonra ayrı bir terminalda `python chatbot.py` əmrini işə salın. Modelinizlə ilk söhbətinizi edin!
