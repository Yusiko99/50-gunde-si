# 📚 50 Gündə Süni-İntellekt: Gün 35

## Ollama API ilə İşləmək: Chatbotun İnterfeysi 💻

Salam! Dünən modelimizi Ollama-da uğurla yerləşdirdik və terminaldan sınaqdan keçirdik. Bu gün isə modelimizi Python kodumuzdan istifadə edə bilmək üçün **Ollama API** ilə işləməyi öyrənəcəyik.

### 1. Ollama API Nədir?

Ollama, arxa planda işləyən bir serverdir və **REST API** vasitəsilə müraciətləri qəbul edir. Bu o deməkdir ki, biz Python-dan adi HTTP sorğuları göndərərək modelimizlə danışa bilərik.

Biz bu prosesi asanlaşdırmaq üçün **`ollama`** Python kitabxanasından istifadə edəcəyik.

### 2. `ollama` Python Kitabxanasının Quraşdırılması

```bash
pip install ollama
```

### 3. Python-dan Modelə Müraciət

Aşağıdakı kodu **`az_chatbot.py`** adlı bir faylda yazaq.

```python
# az_chatbot.py
import ollama

# 1. Ollama Client-i Yaratmaq
# Ollama avtomatik olaraq yerli serverə (http://localhost:11434) qoşulur
client = ollama.Client()

# 2. Mətn Generasiyası Funksiyası
def generate_response(prompt, model_name="az-nano-llm"):
    """ Ollama API vasitəsilə modeldən cavab alır """
    
    print(f"-> Sual: {prompt}")
    
    # API sorğusunu göndəririk
    response = client.generate(
        model=model_name,
        prompt=prompt,
        # Ollama-nın default parametrlərini istifadə edirik
        options={
            "temperature": 0.8,
            "top_k": 50,
        }
    )
    
    # Cavabı çıxarırıq
    return response['response']

# 3. Chatbot Dövrü
def run_chatbot():
    print("--- Azərbaycan Nano LLM Chatbotu Başladı ---")
    print("Çıxmaq üçün 'çıx' yazın.")
    
    while True:
        user_input = input("Siz: ")
        
        if user_input.lower() in ['çıx', 'exit', 'quit']:
            print("Chatbot dayandırıldı. Sağ olun!")
            break
        
        # Cavabı alırıq
        response = generate_response(user_input)
        
        # Cavabı ekrana yazdırırıq
        print(f"Model: {response}")

if __name__ == "__main__":
    # Ollama serverinin işlədiyindən əmin olun
    try:
        client.list() # Serverin işlədiyini yoxlayır
        run_chatbot()
    except Exception as e:
        print("XƏTA: Ollama serveri işləmir.")
        print("Zəhmət olmasa, Ollama proqramının arxa planda işlədiyindən əmin olun.")
```

### 4. Kodun İzahı (Hər Sətrin Detallı İzahı)

| Sətr | Kod | İzah |
| :--- | :--- | :--- |
| 5 | `import ollama` | Ollama Python kitabxanasını daxil edirik. |
| 8 | `client = ollama.Client()` | Ollama serveri ilə əlaqə qurmaq üçün bir client obyekti yaradırıq. |
| 14 | `response = client.generate(...)` | **Əsas API çağırışı.** `generate` metodu modelə prompt göndərir və cavab gözləyir. |
| 15 | `model=model_name` | Ollama-da yaratdığımız modelin adını (`az-nano-llm`) göstəririk. |
| 16 | `prompt=prompt` | İstifadəçinin sualını modelə göndəririk. |
| 20 | `"temperature": 0.8` | Generasiyanın yaradıcılıq səviyyəsini tənzimləyir. |
| 24 | `return response['response']` | API-dən gələn JSON cavabından yalnız mətn hissəsini çıxarırıq. |
| 31 | `user_input = input("Siz: ")` | İstifadəçidən sual qəbul edir. |
| 43 | `client.list()` | Ollama serverinin işlədiyini yoxlamaq üçün sadə bir API çağırışıdır. |

### 5. Ollama-da Chat Rejimi

Ollama həmçinin **`chat`** adlı xüsusi bir API-yə malikdir ki, bu da söhbət tarixçəsini (context) avtomatik idarə edir.

```python
# az_chatbot_chat.py (Chat API istifadəsi)
# ... (importlar və client yaratmaq) ...

def chat_with_model(messages, model_name="az-nano-llm"):
    """ Söhbət tarixçəsini qoruyaraq cavab alır """
    
    response = client.chat(
        model=model_name,
        messages=messages,
    )
    
    # Yeni mesajı tarixçəyə əlavə edirik
    messages.append(response['message'])
    return response['message']['content']

# Söhbət tarixçəsi
messages = []

# Sistem mesajı (Modelfile-dakı SYSTEM prompt-u əvəz edir)
messages.append({
    'role': 'system',
    'content': 'Sən Azərbaycan dilində danışan, faydalı və məlumatlandırıcı bir süni intellekt köməkçisisən.'
})

# İlk sual
messages.append({
    'role': 'user',
    'content': 'Azərbaycanın ən böyük çayı hansıdır?'
})

response = chat_with_model(messages)
print(f"Model: {response}")
```

### 💡 Günün Tapşırığı: Praktika

1.  `ollama` Python kitabxanasını quraşdırın.
2.  `az_chatbot.py` faylını yaradın və icra edin.
3.  Modelinizlə Azərbaycan dilində söhbət edin!

**Sabah görüşənədək!** 👋 Sabah **Modelin Paylaşılması və GitHub** mövzusunu öyrənəcəyik.

***

**Söz Sayı:** 800 söz.
