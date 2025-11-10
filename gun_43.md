# Gün 43: Təlimin Xərcləri və Resursların İdarə Edilməsi 💰

## 43.1. Resursların İdarə Edilməsi

Siz bu layihəni öz kompüterinizdə (RTX 2050) həyata keçirdiniz. Bu, xərcləri minimuma endirdi. Lakin daha böyük modellər üçün bulud xidmətlərindən (AWS, Google Cloud, Azure) istifadə etmək lazım gəlir.

**Resursların Əsas Komponentləri:**

1.  **Hesablama Gücü (Compute):** GPU-nun özü və onun işləmə müddəti.
2.  **Yaddaş (Storage):** Korpusun, Checkpoint-lərin və yekun modelin saxlanması.
3.  **Enerji:** Təlim zamanı sərf olunan elektrik enerjisi.

## 43.2. Təlim Xərclərinin Hesablanması

Bizim 134M parametrli modelimiz üçün xərc hesablaması:

| Parametr | Dəyər | İzahı |
| :--- | :--- | :--- |
| **Model Ölçüsü** | 134 M | Parametrlərin sayı. |
| **Təlim Tokeni** | 1 Milyard | Təlim üçün istifadə olunan ümumi token sayı (korpusun 10 dəfə oxunması). |
| **Təlim Müddəti** | Təxminən 5-7 gün | RTX 2050 (4GB VRAM) üzərində davamlı təlim. |
| **Enerji Sərfiyyatı** | Təxminən 100-150 Watt/saat | RTX 2050-nin orta enerji sərfiyyatı. |

**Bulud Xərcləri (Müqayisə üçün):**

Əgər bu modeli buludda **NVIDIA T4 (16GB VRAM)** GPU-da təlim etsəydiniz:

*   **Təlim Müddəti:** Təxminən 1-2 gün (daha böyük Batch Size sayəsində).
*   **Saatlıq Qiymət:** Təxminən $0.50 - $0.70/saat.
*   **Ümumi Xərc:** $0.70/saat $\times$ 48 saat $\approx$ **$33.60**.

**Nəticə:** Öz kompüterinizdə təlim etmək (enerji xərcləri istisna olmaqla) pulsuzdur, lakin vaxt baxımından daha uzundur.

## 43.3. Resursların Optimallaşdırılması

RTX 2050-də təlim edərkən bu qaydalara əməl edin:

1.  **VRAM-ı Boşaltmaq:** Təlimdən əvvəl bütün lazımsız proqramları (brauzer, oyunlar) bağlayın.
2.  **`torch.cuda.empty_cache()`:** Hər epoxadan sonra PyTorch-un yaddaşını təmizləyin.
3.  **Kiçik Batch Size:** Həmişə ən kiçik Batch Size ilə başlayın və OOM xətası almayana qədər yavaş-yavaş artırın.

**Gündəlik Tapşırıq:** Təlim zamanı kompüterinizin enerji sərfiyyatını və GPU-nun temperaturunu izləyin. Bu məlumatları `TRAINING.md` faylına əlavə edin.
