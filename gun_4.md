# Gün 4: GPU Sürətləndirilməsi: Məhdud Resurslar üçün Optimallaşdırma ⚡

## 4.1. Hesablama Gücünün Paralelləşdirilməsi

Böyük Dil Modellərinin təlimi, yüz minlərlə matris əməliyyatının icrasını tələb edir. Mərkəzi Prosessorlar (CPU) bu əməliyyatları ardıcıl şəkildə yerinə yetirdiyi halda, Qrafik Prosessorları (GPU), xüsusilə **NVIDIA-nın CUDA** (Compute Unified Device Architecture) platforması sayəsində, minlərlə hesablama nüvəsini paralel şəkildə işə salaraq təlim müddətini kəskin şəkildə azaldır.

**Məntiq:** LLM təlimi yüksək dərəcədə paralelləşdirilə bilən bir prosesdir. GPU-nun arxitekturası bu paralelləşdirməyə ideal cavab verir.

## 4.2. PyTorch və CUDA Mühitinin Qurulması

PyTorch-un GPU-dan effektiv istifadə edə bilməsi üçün uyğun CUDA versiyası ilə quraşdırılması zəruridir.

**Quraşdırma Addımları:**

1.  **NVIDIA Sürücüləri:** Əməliyyat sistemində GPU üçün ən son rəsmi sürücülərin quraşdırılması təmin edilməlidir.
2.  **PyTorch Quraşdırılması:** PyTorch-un rəsmi saytından sistemdə mövcud olan CUDA versiyasına uyğun gələn quraşdırma əmri istifadə edilməlidir.

```bash
# Nümunə: PyTorch-un CUDA dəstəkli versiyasını quraşdırmaq
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Hugging Face-in optimallaşdırma kitabxanalarını quraşdırmaq
pip install accelerate transformers datasets
```

## 4.3. Məhdud VRAM (4GB) üçün Kritik Optimallaşdırmalar

100M parametrli modelin 4GB VRAM-da təlimi, yaddaşın idarə edilməsinə xüsusi diqqət yetirməyi tələb edir. Bu, modelin **VRAM tələbini azaltmaq** və **effektiv Batch Size-ı artırmaq** üçün iki əsas texnikanın tətbiqi ilə həyata keçirilir.

### A. Mixed Precision (FP16)

| Xüsusiyyət | Məntiqi Əsas |
| :--- | :--- |
| **Nədir?** | Modelin çəkilərini və qradiyentlərini 32-bit (FP32) əvəzinə **16-bit (FP16)** dəqiqlikdə saxlamaq. |
| **Faydası** | **VRAM istifadəsini 50% azaldır.** Eyni zamanda, müasir GPU-lar (məsələn, T4, RTX seriyası) FP16 əməliyyatlarını daha sürətli icra edə bilir. |
| **Tətbiqi** | PyTorch-da `torch.cuda.amp` (Automatic Mixed Precision) və ya `accelerate` kitabxanası ilə tətbiq olunur. Bu, dəqiqlik itkisini minimuma endirmək üçün əsas hesablamaları FP16-da, kritik hesablamaları isə FP32-də saxlayır. |

### B. Gradient Accumulation (Qradiyent Yığımı)

| Xüsusiyyət | Məntiqi Əsas |
| :--- | :--- |
| **Nədir?** | Modelin çəkilərini yeniləmədən əvvəl bir neçə kiçik Batch-in qradiyentlərini toplamaq. |
| **Faydası** | **Effektiv Batch Size-ı artırır.** Təlimin keyfiyyəti böyük Batch Size-dan asılıdır. Məhdud VRAM böyük Batch Size-a icazə vermədikdə, bu texnika kiçik Batch-lərin qradiyentlərini toplayaraq, böyük Batch Size-ın təsirini simulyasiya edir. |
| **Tətbiqi** | Təlim dövründə `loss.backward()` əməliyyatını `N` dəfə icra edib, `optimizer.step()` əməliyyatını yalnız `N`-ci dəfədən sonra yerinə yetirməklə həyata keçirilir. |

Bu iki texnika, məhdud VRAM şəraitində LLM təliminin **imkanlılığını** və **effektivliyini** təmin edən əsas sütunlardır.
