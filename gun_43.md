# Gün 43: Təlimin Xərcləri və Resursların İdarə Edilməsi 💰

## 43.1. Resursların Təhlili

LLM təlimi, hesablama gücü (Compute) və yaddaş (VRAM/RAM) baxımından ən bahalı Sİ tapşırıqlarından biridir.

**Əsas Resurs Komponentləri:**

1.  **Hesablama Gücü (GPU):** Təlimin sürətini və mümkün olan model ölçüsünü müəyyənləşdirir.
2.  **VRAM:** Modelin çəkilərini, qradiyentlərini və aralıq hesablamaları saxlamaq üçün istifadə olunur.
3.  **Enerji:** Təlim zamanı sərf olunan elektrik enerjisi.

## 43.2. Məhdud Resurslarda Xərc Effektivliyi

Bu layihənin məntiqi əsası, məhdud resurslarda (4GB VRAM) LLM təliminin necə həyata keçirilməsini göstərməkdir.

| Resurs | Təlim Müddəti | Məntiqi Əsas |
| :--- | :--- | :--- |
| **RTX 2050 (4GB VRAM)** | Təxminən 5-7 gün | **Xərc Effektivliyi:** Bulud xidmətlərindən istifadə etmədən, yalnız enerji xərcləri ilə təlimi həyata keçirmək. |
| **NVIDIA T4 (Bulud)** | Təxminən 1-2 gün | **Sürət:** Daha böyük VRAM (16GB) və daha yüksək hesablama gücü sayəsində daha böyük Batch Size istifadə etmək və təlimi sürətləndirmək. |

**Məntiq:** Təlimin xərci **Vaxt** və **Pul** arasında bir kompromisdir. Məhdud resurslarda təlim vaxtı uzadır, lakin pul xərcini minimuma endirir.

## 43.3. Resursların Optimallaşdırılması

Məhdud VRAM-da təlim edərkən tətbiq edilən əsas optimallaşdırma prinsipləri:

1.  **FP16 (Mixed Precision):** VRAM istifadəsini 50% azaltmaq.
2.  **Gradient Accumulation:** Kiçik Batch Size ilə böyük Batch Size-ın təsirini simulyasiya etmək.
3.  **VRAM Təmizlənməsi:** Təlim dövründə lazımsız tensorları silmək üçün `torch.cuda.empty_cache()` funksiyasından istifadə etmək.

**Nəticə:** Resursların idarə edilməsi, LLM tərtibatçısının ən vacib bacarıqlarından biridir. Modelin ölçüsü və təlimin müddəti mövcud resurslara uyğun olaraq diqqətlə planlaşdırılmalıdır.
