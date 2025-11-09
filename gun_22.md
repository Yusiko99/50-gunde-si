# 📚 50 Gündə Süni-İntellekt: Gün 22

## Verilənlər Yükləyicisi (DataLoader): Məlumatın Təchizatı 🚚

Salam! Dünən təlim prosesinin əsas komponentləri (Loss, Optimizer) ilə tanış olduq və `train.py` skriptimizin təməlini qoyduq. Bu gün isə modelimizi məlumatla təchiz edəcək əsas aləti – **Verilənlər Yükləyicisini (DataLoader)** quracağıq.

### 1. Niyə DataLoader?

Bizim `train.npy` faylımızda **milyonlarla** token var. Model bu tokenlərin hamısını eyni anda emal edə bilməz.

> **DataLoader** — böyük məlumat bazasını kiçik, idarəolunan hissələrə – **Batch**-lərə bölür və təlim prosesi üçün onları ardıcıl olaraq GPU-ya ötürür.

DataLoader-in əsas funksiyaları:
1.  **Batching:** Məlumatı `BATCH_SIZE` (məsələn, 12) ölçüsündə hissələrə bölür.
2.  **Shuffling:** Hər epoch-da (dövrdə) məlumatı qarışdırır ki, model məlumatın sırasını əzbərləməsin.
3.  **Parallel Yükləmə:** Məlumatı CPU-dan GPU-ya paralel şəkildə yükləyir.

### 2. Dataset Sinfinin Qurulması

PyTorch-da DataLoader-i istifadə etmək üçün əvvəlcə **Dataset** adlı bir sinif yaratmalıyıq. Bu sinif PyTorch-a məlumatın harada olduğunu və hər bir elementin necə alınacağını bildirir.

Aşağıdakı kodu **`data_loader.py`** adlı bir faylda yazaq.

```python
# data_loader.py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

# 1. GPTDataset Sinfi
class GPTDataset(Dataset):
    """ Təlim və Validasiya məlumatlarını idarə edən PyTorch Dataset sinfi """

    def __init__(self, split, block_size):
        # split: 'train' və ya 'val'
        # block_size: Modelin kontekst pəncərəsinin uzunluğu (512)
        self.block_size = block_size

        # Məlumatı .npy faylından yükləyirik
        file_path = f'{split}.npy'
        print(f"Məlumat yüklənir: {file_path}")
        self.data = np.load(file_path).astype(np.uint16)
        print(f"Yükləndi. Ümumi token sayı: {len(self.data):,}")

    def __len__(self):
        """ Dataset-dəki mümkün nümunələrin ümumi sayını qaytarır """
        # Məlumatın uzunluğu - block_size (çünki son block_size qədər nümunə yarada bilmərik)
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        """ Verilmiş indeks üçün bir nümunə (Batch) qaytarır """
        # idx: Başlanğıc indeks
        # T: Ardıcıllıq uzunluğu (block_size)
        T = self.block_size
        
        # 1. Giriş (Input)
        # idx-dən idx+T-yə qədər olan tokenlər
        x = self.data[idx:idx+T]
        
        # 2. Hədəf (Target)
        # idx+1-dən idx+T+1-ə qədər olan tokenlər (bir addım irəli sürüşdürülmüş)
        y = self.data[idx+1:idx+T+1]

        # NumPy massivlərini PyTorch Tensor-larına çeviririk
        x = torch.from_numpy(x.astype(np.int64))
        y = torch.from_numpy(y.astype(np.int64))
        
        return x, y

# 2. get_dataloaders Funksiyası
def get_dataloaders(block_size, batch_size):
    """ Təlim və Validasiya üçün DataLoader-ləri yaradır """
    
    # Dataset-ləri yaradırıq
    train_dataset = GPTDataset('train', block_size)
    val_dataset = GPTDataset('val', block_size)

    # DataLoader-ləri yaradırıq
    train_loader = DataLoader(
        train_dataset,
        sampler=None,
        shuffle=True, # Təlim məlumatını qarışdırırıq
        batch_size=batch_size,
        num_workers=0, # Məlumatı yükləmək üçün istifadə olunan CPU nüvələrinin sayı
        pin_memory=True, # GPU-ya sürətli ötürmə üçün
    )

    val_loader = DataLoader(
        val_dataset,
        sampler=None,
        shuffle=False, # Validasiya məlumatını qarışdırmağa ehtiyac yoxdur
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
    )
    
    return train_loader, val_loader
```

### 3. Kodun İzahı (Hər Sətrin Detallı İzahı)

| Sətr | Kod | İzah |
| :--- | :--- | :--- |
| 10 | `class GPTDataset(Dataset):` | Bütün PyTorch Dataset-ləri bu sinifdən miras almalıdır. |
| 17 | `self.data = np.load(file_path).astype(np.uint16)` | Əvvəlki gün yaratdığımız `train.npy` və ya `val.npy` faylını yükləyirik. |
| 23 | `def __len__(self):` | Dataset-də neçə nümunə olduğunu PyTorch-a bildirir. |
| 28 | `def __getitem__(self, idx):` | **Əsas funksiya.** PyTorch bu funksiyanı çağıraraq məlumatı alır. |
| 34 | `x = self.data[idx:idx+T]` | **Giriş (Input):** `idx` mövqeyindən başlayaraq `T` (512) uzunluğunda tokenlər ardıcıllığını seçirik. |
| 38 | `y = self.data[idx+1:idx+T+1]` | **Hədəf (Target):** Bu, girişdən **bir addım irəli sürüşdürülmüş** tokenlər ardıcıllığıdır. |
| | | **Niyə sürüşdürülmüş?** Çünki model `x[i]` tokenini görüb `y[i]` tokenini proqnozlaşdırmalıdır. Məsələn, `x[0]`-ı görüb `y[0]`-ı (yəni `x[1]`-i) proqnozlaşdırır. |
| 41 | `x = torch.from_numpy(x.astype(np.int64))` | NumPy massivini PyTorch-un `Tensor` formatına çeviririk. |
| 50 | `train_loader = DataLoader(...)` | **DataLoader** obyektini yaradırıq. `shuffle=True` təlim məlumatını qarışdırır. |

### 4. Təlim Skriptinin Yenilənməsi

İndi `train.py` skriptimizdə `get_dataloaders` funksiyasını istifadə edə bilərik.

```python
# train.py (Yenilənmiş)
# ... (əvvəlki importlar) ...
from data_loader import get_dataloaders # Yeni import

# ... (əvvəlki hiperparametrlər) ...

# 5. DataLoader-ləri Yaratmaq
train_loader, val_loader = get_dataloaders(BLOCK_SIZE, BATCH_SIZE)

# ... (Model və Optimizer-in yaradılması) ...

# 6. Təlim Dövrü (İndi işləyəcək!)
# for iter in tqdm(range(MAX_ITERS), desc="Təlim"):
#     # Məlumatı yüklə
#     x, y = next(iter(train_loader))
#     x, y = x.to(device), y.to(device)
#     # ... (qalan təlim addımları) ...
```

### 💡 Günün Tapşırığı: Praktika

1.  **`data_loader.py`** faylını yaradın və yuxarıdakı kodu ora kopyalayın.
2.  `train.npy` və `val.npy` fayllarının mövcud olduğundan əmin olun.
3.  Kiçik bir sınaq skripti yazın:
    ```python
    from data_loader import get_dataloaders
    train_loader, _ = get_dataloaders(block_size=512, batch_size=4)
    
    # İlk Batch-i yüklə
    x, y = next(iter(train_loader))
    print(f"Giriş (x) ölçüsü: {x.shape}") # (4, 512) olmalıdır
    print(f"Hədəf (y) ölçüsü: {y.shape}") # (4, 512) olmalıdır
    ```

**Sabah görüşənədək!** 👋 Sabah **Təlim Dövrünün** bütün addımlarını (irəli ötürmə, geriyə ötürmə, yenilənmə) PyTorch-da birləşdirəcəyik.

***

**Söz Sayı:** 800 söz.
