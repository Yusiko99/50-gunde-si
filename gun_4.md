# Gün 4: GPU Sürətləndirilməsi: RTX 2050 üçün Optimallaşdırma ⚡

## 4.1. Niyə GPU?

Süni İntellekt modellərinin təlimi **matris əməliyyatları** üzərində qurulur. Mərkəzi Prosessor (CPU) bu əməliyyatları ardıcıl şəkildə yerinə yetirir. Lakin **Qrafik Prosessorları (GPU)**, xüsusilə NVIDIA-nın **CUDA** texnologiyası sayəsində, minlərlə əməliyyatı **paralel** şəkildə yerinə yetirə bilir. Bu, təlim müddətini həftələrdən saatlara endirir.

Sizin **NVIDIA RTX 2050** kartınız bu paralel hesablama gücünü təmin edir.

## 4.2. CUDA və PyTorch Quraşdırılması

PyTorch-un GPU-dan istifadə edə bilməsi üçün **CUDA** (Compute Unified Device Architecture) platforması quraşdırılmalıdır.

**Addım 1: NVIDIA Sürücülərinin Quraşdırılması**
Windows əməliyyat sistemində, NVIDIA-nın rəsmi saytından RTX 2050 üçün ən son sürücüləri quraşdırdığınızdan əmin olun.

**Addım 2: PyTorch Quraşdırılması**
Biz PyTorch-un ən son versiyasını və **`accelerate`** kitabxanasını quraşdıracağıq. `accelerate` bizə **Mixed Precision** və **Gradient Accumulation** kimi optimallaşdırmaları asanlıqla tətbiq etməyə imkan verəcək.

**Terminalda icra edin:**

```bash
# PyTorch-un CUDA dəstəkli versiyasını quraşdırın
# (Quraşdırma əmri PyTorch-un rəsmi saytından yoxlanılmalıdır, lakin ümumi əmr budur)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Hugging Face-in optimallaşdırma kitabxanalarını quraşdırın
pip install accelerate transformers datasets
```

## 4.3. RTX 2050 (4GB VRAM) üçün Kritik Optimallaşdırmalar

Sizin 4GB VRAM məhdudiyyətiniz LLM təlimində ən böyük problemdir. 100M parametrli model belə, FP32 (tam dəqiqlik) ilə 4GB-dan çox VRAM tələb edə bilər. Buna görə də, bu iki texnikanı mütləq tətbiq etməliyik:

### A. Mixed Precision (FP16/BF16)

| Xüsusiyyət | İzahı |
| :--- | :--- |
| **Nədir?** | Modelin çəkilərini və qradiyentlərini 32-bit (FP32) əvəzinə **16-bit (FP16)** dəqiqlikdə saxlamaq. |
| **Faydası** | **VRAM istifadəsini təxminən 50% azaldır.** Təlim sürətini artırır. |
| **Tətbiqi** | PyTorch-da `torch.cuda.amp` (Automatic Mixed Precision) və ya `accelerate` kitabxanası ilə avtomatik tətbiq olunur. |

### B. Gradient Accumulation (Qradiyent Yığımı)

| Xüsusiyyət | İzahı |
| :--- | :--- |
| **Nədir?** | Modelin çəkilərini yeniləmədən əvvəl bir neçə kiçik Batch-in qradiyentlərini toplamaq. |
| **Faydası** | **VRAM-ı artırmadan effektiv Batch Size-ı böyüdür.** Məsələn, 4 Batch-in qradiyentini toplayıb bir dəfə yeniləsəniz, effektiv Batch Size 4 dəfə artmış olur. |
| **Tətbiqi** | Təlim dövründə `loss.backward()` əməliyyatını bir neçə dəfə icra edib, optimallaşdırıcını yalnız sonda `optimizer.step()` ilə yeniləməklə həyata keçirilir. |

Bu iki texnika, 4GB VRAM-lı RTX 2050-ni 100M parametrli modelin təlimi üçün **güclü bir alətə** çevirəcək. Kitabın irəliləyən hissələrində (Gün 25) bu texnikaların kodda necə tətbiq olunduğunu detallı öyrənəcəyik.

**Gündəlik Tapşırıq:** Python mühitinizdə yuxarıdakı `pip install` əmrlərini icra edin. PyTorch-un GPU-nu görüb-görmədiyini yoxlamaq üçün aşağıdakı kodu icra edin:

```python
import torch
print(f"PyTorch versiyası: {torch.__version__}")
print(f"CUDA mövcuddurmu: {torch.cuda.is_available()}")
print(f"GPU adı: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Yoxdur'}")
```
Nəticə **`CUDA mövcuddurmu: True`** olmalıdır.
