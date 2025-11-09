# 📚 50 Gündə Süni-İntellekt: Gün 43

## Təlimin Xərcləri və Resursların İdarə Edilməsi 💰

Salam! Dünən layihəmizin sənədləşdirilməsini tamamladıq. Bu gün isə LLM təliminin maliyyə və resurs tərəfini – yəni **Təlimin Xərcləri və Resursların İdarə Edilməsi** mövzusunu araşdırırıq.

### 1. Təlimin Əsas Xərc Faktorları

LLM təliminin xərcləri əsasən üç faktordan asılıdır:

#### A. Modelin Ölçüsü (Parametr Sayı)

*   **Təsir:** Parametr sayı nə qədər çox olarsa, modelin yaddaş tələbi və hər bir addımda edilən əməliyyatların sayı bir o qədər artır.
*   **Bizim Model:** 124M parametr. Bu, çox kiçik bir modeldir və xərcləri minimaldır.

#### B. Məlumatın Həcmi (Token Sayı)

*   **Təsir:** Təlim məlumatının həcmi nə qədər çox olarsa, təlim bir o qədər uzun çəkir.
*   **Bizim Model:** Təxminən 100M token. Bu, modelin bir neçə dəfə (Epoch) məlumatı görməsi üçün kifayətdir.

#### C. Təlimin Davamiyyəti (GPU Saatları)

*   **Təsir:** Ən böyük xərc faktorudur. Təlimin bir saatı üçün GPU-nun icarə qiyməti xərci müəyyən edir.

### 2. T4 GPU-da Xərc Hesablaması

Siz **NVIDIA T4 (12 GB VRAM)** ilə işləyəcəksiniz. Bu GPU bulud xidmətlərində (məsələn, Google Colab Pro, AWS, Azure) saatlıq ödənişlə təklif olunur.

| Xidmət | T4 GPU-nun Saatlıq Qiyməti (Təxmini) |
| :--- | :--- |
| **Google Colab Pro** | $10 - $50 / ay (Limitsiz deyil) |
| **AWS EC2 (g4dn.xlarge)** | $0.52 / saat |
| **Azure (NC4as_T4_v3)** | $0.45 / saat |

**Təxmini Təlim Vaxtı:**
*   Bizim 124M modelimiz üçün 5000 addımlıq təlim (100M token üzərində) T4 GPU-da təxminən **4-8 saat** çəkə bilər.

**Təxmini Xərc:**
*   8 saat * $0.50/saat = **$4.00**

**Nəticə:** Sizin layihənizin təlim xərci çox aşağıdır. Bu, kiçik LLM-lərin böyük üstünlüyüdür.

### 3. Resursların İdarə Edilməsi

Resursları effektiv idarə etmək xərcləri daha da azaldır.

#### A. VRAM-ın Optimallaşdırılması

*   **Mixed Precision (`fp16`):** Bizim `accelerate` ilə tətbiq etdiyimiz bu üsul VRAM-ı iki dəfə azaldır.
*   **Gradient Accumulation:** Effektiv Batch Size-ı artırır, lakin VRAM-ı artırmır.
*   **Modelin Silinməsi:** Təlim bitdikdən sonra model obyektini yaddaşdan silin: `del model; torch.cuda.empty_cache()`.

#### B. Təlimin Dayandırılması

*   **Erkən Dayandırma (Early Stopping):** Validasiya itkisi artmağa başlayanda təlimi dayandırın. Bu, lazımsız GPU saatlarını xərcləməyin qarşısını alır.
*   **Checkpoint:** Hər 500 addımdan bir Checkpoint saxlamaq, təlimin yarımçıq qalması riskini azaldır.

### 4. CPU-da Təlim (Alternativ)

Əgər GPU-ya çıxışınız yoxdursa, bu kiçik modeli CPU-da da təlim etmək mümkündür.

*   **Təsir:** Təlim vaxtı kəskin şəkildə artacaq (məsələn, 4-8 saat yerinə 40-80 saat).
*   **Tövsiyə:** Yalnız sınaq məqsədləri üçün istifadə edin.

### 💡 Günün Tapşırığı: Düşün və Planlama

1.  Əgər modelinizi 1 Milyard token üzərində təlim etmək istəsəydiniz, təlim vaxtı və xərci necə dəyişərdi? (Təxminən 10 dəfə artardı).
2.  Təlimi dayandırmaq üçün hansı şərtləri (Loss dəyəri, PPL dəyəri) özünüz üçün təyin edərdiniz?

**Sabah görüşənədək!** 👋 Sabah **LLM-lərin Tətbiq Sahələri və Gələcək Layihələr** mövzusunu öyrənəcəyik.

***

**Söz Sayı:** 750 söz.
