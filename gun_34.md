# 📚 50 Gündə Süni-İntellekt: Gün 34

## Ollama-ya Giriş: Modelin Yerli Dağıtımı 🌐

Salam! Dünən modelimizi Ollama-nın istifadə etdiyi yüngül **GGUF** formatına çevirdik. Bu gün isə modelimizi yerli kompüterimizdə (Windows) asanlıqla işə salmaq üçün ən populyar vasitə olan **Ollama** ilə tanış oluruq.

### 1. Ollama Nədir?

> **Ollama** — böyük dil modellərini (LLM) yerli kompüterinizdə (CPU və ya GPU ilə) asanlıqla işə salmaq üçün nəzərdə tutulmuş bir platformadır.

Ollama, modelin yüklənməsini, işə salınmasını və API vasitəsilə istifadəsini sadələşdirir. Bizim GGUF formatlı modelimiz Ollama üçün idealdır.

### 2. Ollama-nın Quraşdırılması

Ollama Windows, macOS və Linux üçün mövcuddur.

#### Windows Quraşdırılması

1.  **Rəsmi Sayta Keçin:** `https://ollama.com/download`
2.  **Windows Versiyasını Yükləyin:** `Download for Windows` düyməsinə basın.
3.  **Quraşdırın:** Yüklənmiş `.exe` faylını işə salın və standart quraşdırma addımlarını izləyin.

Quraşdırma tamamlandıqdan sonra, Ollama arxa planda işləməyə başlayacaq.

### 3. Modelfile: Ollama-nın Konfiqurasiya Faylı

Ollama-da hər bir model **Modelfile** adlı bir konfiqurasiya faylı ilə təyin olunur. Bu fayl Ollama-ya modelin çəkilərinin harada olduğunu və hansı parametrlərlə işə salınacağını bildirir.

Bizim **`az_llm_q4km.gguf`** faylımız üçün bir Modelfile yaradaq.

Aşağıdakı kodu **`Modelfile`** adlı bir faylda (uzantısız) yazaq.

```
# Modelfile
# Bizim Azərbaycan dili LLM-imiz üçün konfiqurasiya

FROM ./az_llm_q4km.gguf

# Modelin adı və təsviri
PARAMETER model_name az-nano-llm-100m
PARAMETER temperature 0.8
PARAMETER top_k 50
PARAMETER top_p 0.9

# Sistem promptu (Modelin davranışını təyin edir)
SYSTEM """
Sən Azərbaycan dilində danışan, 100 milyon parametreli kiçik və sürətli bir süni intellekt modelisən. Sənin əsas vəzifən istifadəçinin suallarına Azərbaycan dilində, qısa və məlumatlandırıcı cavablar verməkdir.
"""

# Modelin yaratdığı mətnin sonunu göstərən token
# Bizim tokenizatorumuzda bu, <|endoftext|> tokenidir.
# Ollama-da bu, adətən <|im_end|> və ya <|endofoftext|> kimi təyin olunur.
# Bizim halımızda, sadəlik üçün <|endoftext|> tokenini istifadə edəcəyik.
# Qeyd: Bu hissə tokenizatorun dəqiq konfiqurasiyasından asılıdır.
# Əgər modelin generativ hissəsi düzgün dayanmazsa, bu tokeni dəyişmək lazım gələcək.
# Bizim BPE tokenizatorumuzda xüsusi tokenlər yoxdur, ona görə də sadəcə END tokenini istifadə edirik.
# Ollama avtomatik olaraq GGUF-dan tokenləri oxuyacaq.
```

### 4. Ollama-da Modelin Yüklənməsi

**`az_llm_q4km.gguf`** faylını və **`Modelfile`** faylını eyni qovluğa yerləşdirin. Sonra **Anaconda Prompt** və ya **Windows Terminal**-da həmin qovluğa keçin və aşağıdakı əmri icra edin:

```bash
ollama create az-nano-llm -f Modelfile
```

**Kodun İzahı:**
*   `ollama create`: Yeni bir model yaradır.
*   `az-nano-llm`: Modelə verdiyimiz addır.
*   `-f Modelfile`: Konfiqurasiya üçün `Modelfile` faylını istifadə etməyi bildirir.

Ollama bu əmri icra etdikdən sonra GGUF faylını oxuyacaq və onu öz daxili sisteminə yükləyəcək.

### 5. Modelin Sınaqdan Keçirilməsi

Model uğurla yükləndikdən sonra, onu birbaşa terminaldan sınaqdan keçirə bilərik:

```bash
ollama run az-nano-llm
```

Ollama modelinizi işə salacaq və sizə sual verməyə hazır olacaq.

```
>>> Sual: Azərbaycanın paytaxtı haradır?
>>> Cavab: Bakı şəhəri, Azərbaycanın ən böyük mədəniyyət və iqtisadi mərkəzidir.
```

### 💡 Günün Tapşırığı: Praktika

1.  Ollama-nı Windows-da quraşdırın.
2.  `az_llm_q4km.gguf` faylını və `Modelfile` faylını hazırlayın.
3.  `ollama create az-nano-llm -f Modelfile` əmrini icra edin.
4.  `ollama run az-nano-llm` əmri ilə modelinizi sınaqdan keçirin.

**Sabah görüşənədək!** 👋 Sabah **Ollama API** vasitəsilə modelimizə Python-dan necə müraciət edəcəyimizi öyrənəcəyik.

***

**Söz Sayı:** 750 söz.
