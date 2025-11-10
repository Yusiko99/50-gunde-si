# 📚 50 Gündə Süni-İntellekt: Gün 4

## GPU Sürətləndirilməsi: CUDA və PyTorch ⚡

Salam! Üçüncü gündə virtual mühitimizi qurduq. Bu gün isə LLM layihəmizin ən vacib hissələrindən birinə – **GPU Sürətləndirilməsinə** keçirik.

### 1. CPU vs. GPU: Niyə GPU?

**CPU (Central Processing Unit)** – Kompüterin beynidir. O, ardıcıl, mürəkkəb tapşırıqları sürətlə yerinə yetirmək üçün nəzərdə tutulub.

**GPU (Graphics Processing Unit)** – Əvvəlcə qrafikləri emal etmək üçün yaradılsa da, Dərin Öyrənmədə əvəzolunmazdır. Niyə?

| Xüsusiyyət | CPU | GPU |
| :--- | :--- | :--- |
| **Nüvə Sayı** | Az (4-16) | Çox (Minlərlə) |
| **İş Prinsipi** | Ardıcıl, mürəkkəb | Paralel, sadə |
| **LLM Təlimi** | Çox yavaş (Həftələr) | **Çox sürətli (Saatlar/Günlər)** |

LLM təlimi, eyni anda minlərlə sadə riyazi əməliyyatın (matris vurulması) paralel şəkildə aparılmasını tələb edir. Məhz buna görə də, minlərlə nüvəyə malik olan **GPU** bu işdə CPU-dan **yüzlərlə dəfə** daha sürətlidir.

### 2. CUDA: GPU-nun Dili

Sizin qrafik kartınızın gücünü istifadə etmək üçün bir "tərcüməçi" lazımdır. Bu tərcüməçi **CUDA** adlanır.

> **CUDA** (Compute Unified Device Architecture) — NVIDIA tərəfindən yaradılmış, proqramçıların NVIDIA GPU-ların paralel hesablama gücündən istifadə etməsinə imkan verən bir platformadır.

PyTorch kimi Dərin Öyrənmə kitabxanaları, GPU-ya nə etməli olduğunu məhz CUDA vasitəsilə "deyir".

#### Addım 1: NVIDIA Sürücülərinin Yoxlanılması

Ən son **NVIDIA sürücülərinin** quraşdırıldığına əmin olun.

#### Addım 2: CUDA Toolkit-in Quraşdırılması

PyTorch-u quraşdırarkən, hansı CUDA versiyasını dəstəklədiyini bilməliyik. Ən yaxşı yanaşma, PyTorch-un rəsmi saytında tövsiyə olunan **CUDA Toolkit** versiyasını yükləməkdir.

**Qeyd:** Conda istifadə etdiyimiz üçün, bəzən **CUDA Toolkit-i əməliyyat sisteminə quraşdırmaq əvəzinə**, onu birbaşa Conda mühitinə quraşdırmaq daha asan olur. Biz də bu yoldan istifadə edəcəyik.

### 3. PyTorch-un Quraşdırılması

**PyTorch** bizim LLM-i quracağımız və təlim edəcəyimiz əsas Dərin Öyrənmə çərçivəsidir.

#### Addım 1: Virtual Mühiti Aktivləşdirmək

Əvvəlcə, dünən yaratdığımız virtual mühiti aktivləşdiririk:

```bash
conda activate llm_50gun
```

#### Addım 2: PyTorch-u Quraşdırmaq

PyTorch-un rəsmi saytına daxil olun və **"Install PyTorch"** bölməsində Conda üçün olan əmri kopyalayın. Bu əmr həm PyTorch-u, həm də onun tələb etdiyi **CUDA** kitabxanalarını avtomatik olaraq `llm_50gun` mühitinə quraşdıracaq.

Tipik bir quraşdırma əmri belə görünəcək (versiyalar dəyişə bilər):

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

**İzah:**
*   `conda install`: Conda-ya paketləri quraşdırmasını əmr edir.
*   `pytorch torchvision torchaudio`: Əsas PyTorch kitabxanalarıdır.
*   `pytorch-cuda=12.1`: PyTorch-un **CUDA 12.1** versiyası ilə işləyən versiyasını tələb edir.
*   `-c pytorch -c nvidia`: Paketləri PyTorch və NVIDIA-nın rəsmi kanallarından yükləməsini bildirir.

Bu əmri **Anaconda Prompt** pəncərəsində icra edin. Yükləmə həcmi böyük ola bilər (bir neçə GB).

### 4. Quraşdırmanın Yoxlanılması

Quraşdırma bitdikdən sonra, PyTorch-un GPU-nu görüb-görmədiyini yoxlayaq.

1.  Python interaktiv mühitinə daxil olun:

```bash
python
```

2.  Aşağıdakı Python kodunu sətir-sətir yazın və Enter-ə basın:

```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
```

**Kodun İzahı:**

| Kod Sətri | İzah | Nəticə |
| :--- | :--- | :--- |
| `import torch` | PyTorch kitabxanasını proqramımıza daxil edirik. | |
| `print(torch.__version__)` | PyTorch-un hansı versiyasının quraşdırıldığını ekrana çıxarır. | Məsələn, `2.3.0+cu121` |
| `print(torch.cuda.is_available())` | **Ən vacib sətir!** PyTorch-un kompüterinizdə **CUDA** dəstəkli bir GPU (sizin halınızda T4) tapıb-tapmadığını yoxlayır. | **`True`** (Düzgün quraşdırılıbsa) |

Əgər nəticə **`True`** olarsa, deməli, siz artıq LLM təlimi üçün GPU-nuzun bütün gücündən istifadə etməyə hazırsınız!

### 💡 Günün Tapşırığı: Praktika

1.  `llm_50gun` virtual mühitini aktivləşdirin.
2.  PyTorch-u yuxarıdakı əmrə bənzər şəkildə (ən son versiyaları yoxlayaraq) quraşdırın.
3.  Python interaktiv mühitində `torch.cuda.is_available()` əmrinin **`True`** nəticəsini verdiyinə əmin olun.

**Sabah görüşənədək!** 👋 Sabah Dərin Öyrənmənin təməlini təşkil edən bəzi əsas Python kitabxanaları (`numpy`, `pandas`) ilə tanış olacağıq.

