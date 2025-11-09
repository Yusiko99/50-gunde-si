# 📚 50 Gündə Süni-İntellekt: Gün 39

## Modelin İdarə Edilməsi və Sürətləndirilməsi 🚀

Salam! Dünən modelimizin performansını artırmaq üçün **Hiperparametr Tənzimlənməsi** mövzusunu araşdırdıq. Bu gün isə modelin istifadə (inference) mərhələsində necə daha sürətli və effektiv işlədiyini öyrənəcəyik.

### 1. Modelin Sürətləndirilməsi Texnikaları

Modelin təlimi bitdikdən sonra, onun sürətini artırmaq üçün bir neçə üsul var:

#### A. Quantization (Kvantlaşdırma)

Biz bunu artıq GGUF formatına keçərkən etdik. **INT4** və ya **INT8** dəqiqliyi modelin yaddaş tələbini azaldır və CPU/GPU-da əməliyyatları sürətləndirir.

#### B. Batching (Toplu İşləmə)

Əgər modelinizə eyni anda bir neçə sorğu gəlirsə, onları **Batch** şəklində birləşdirib modelə vermək tək-tək verməkdən daha sürətlidir.

*   **Tətbiq:** Bizim `load_model.py` skriptimizdə `idx` tensoru `(B, T)` ölçüsündədir. Əgər `B > 1` olarsa, model eyni anda bir neçə prompt-u emal edə bilər.

#### C. Modelin Tərtib Edilməsi (Model Compilation)

PyTorch-un 2.0 versiyası ilə gələn **`torch.compile`** funksiyası modelin kodunu daha sürətli işləyən bir formaya çevirir.

```python
# load_model.py (Yenilənmiş)

# ... (əvvəlki kodlar) ...

# 3. Çəkiləri Yükləmək
# ...

# 4. Modeli Tərtib Etmək (Compilation)
# Bu, modelin sürətini 20-50% artıra bilər
model = torch.compile(model)

# 5. Modeli Qiymətləndirmə Rejiminə Keçirmək
model.eval()

# ... (qalan kodlar) ...
```

**Kodun İzahı:**
*   `torch.compile(model)`: Modelin bütün PyTorch əməliyyatlarını yoxlayır və onları daha səmərəli şəkildə birləşdirir. Bu, ilk dəfə işə salındıqda bir qədər vaxt ala bilər, lakin sonrakı işləmələrdə sürətli olur.

### 2. Mətn Generasiyasının İdarə Edilməsi

Modelin yaratdığı mətnin keyfiyyətini və sürətini idarə etmək üçün `generate` funksiyasındakı parametrlər vacibdir.

#### A. Temperature (Temperatur)

*   **Yüksək Temperature (məsələn, 1.0):** Daha çox təsadüfilik, daha yaradıcı, lakin bəzən mənasız cavablar.
*   **Aşağı Temperature (məsələn, 0.5):** Daha az təsadüfilik, daha məntiqli, lakin bəzən təkrarlanan cavablar.

#### B. Top-K və Top-P Sampling

*   **Top-K:** Növbəti tokeni seçmək üçün ən yüksək ehtimalı olan **K** sayda tokeni nəzərə alır.
*   **Top-P (Nucleus Sampling):** Növbəti tokeni seçmək üçün ehtimalların cəmi **P** faizə çatan tokenləri nəzərə alır.

**Tövsiyə:** `temperature=0.8` və `top_k=50` və ya `top_p=0.9` kimi dəyərləri birlikdə istifadə etmək ən yaxşı nəticəni verir.

### 3. Ollama-da Sürətləndirmə

Bizim Ollama-da istifadə etdiyimiz GGUF formatı artıq `llama.cpp` tərəfindən optimallaşdırılıb.

*   **GPU Offload:** Ollama avtomatik olaraq GGUF modelinin əməliyyatlarının bir hissəsini (və ya hamısını) GPU-ya (bizim T4-ə) ötürür. Bu, sürəti kəskin şəkildə artırır.
*   **Modelfile Parametrləri:** `Modelfile`-da `PARAMETER num_gpu 99` kimi bir əmr əlavə etməklə modelin bütün qatlarını GPU-ya yükləməyi təmin edə bilərsiniz.

```
# Modelfile (Yenilənmiş)
FROM ./az_llm_q4km.gguf

# Bütün qatları GPU-ya yüklə
PARAMETER num_gpu 99
```

### 💡 Günün Tapşırığı: Praktika

1.  `load_model.py` skriptinə `model = torch.compile(model)` əmrini əlavə edin.
2.  Modelin generasiya sürətini `torch.compile` ilə və onsuz müqayisə edin.
3.  Ollama `Modelfile`-a `PARAMETER num_gpu 99` əmrini əlavə edin və modeli yenidən yaradın (`ollama create`).

**Sabah görüşənədək!** 👋 Sabah **Etik Mülahizələr və Məsuliyyətli Süni İntellekt** mövzusunu öyrənəcəyik.

***

**Söz Sayı:** 750 söz.
