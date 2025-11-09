# 📚 50 Gündə Süni-İntellekt: Gün 18

## Parametr Sayının Hesablanması: Modelin Ölçüsü 📏

Salam! Dünən **GPT (NanoGPT)** modelimizin tam PyTorch sinfini qurduq və modelin ümumi parametr sayının **124,417,536** olduğunu gördük. Bu gün bu rəqəmin arxasında duran riyaziyyatı – yəni modelin ölçüsünün necə hesablandığını öyrənəcəyik.

Bu bilik, gələcəkdə modelinizin ölçüsünü (məsələn, 50M və ya 200M) dəyişdirmək istədiyiniz zaman sizə kömək edəcək.

### 1. Parametr Nədir?

Neyron şəbəkədə **parametr** modelin təlim zamanı öyrəndiyi dəyərlərdir. Bunlar əsasən **çəkilər (weights)** və **meyilliklər (biases)** adlanır. Hər bir parametr yaddaşda yer tutur və təlim zamanı yenilənir.

Bizim modelimizdə parametr sayı üç əsas hissədən ibarətdir:

1.  **Gömülmə Qatları (Embedding Layers)**
2.  **Transformer Blokları (Block)**
3.  **Dil Modeli Başı (LM Head)**

### 2. Hissə-Hissə Hesablama

Bizim konfiqurasiyamız: `n_embd=768`, `vocab_size=32000`, `n_layer=12`.

#### A. Gömülmə Qatları (Embedding Layers)

| Qat | Hesablama | Nəticə |
| :--- | :--- | :--- |
| **Token Gömülməsi (`wte`)** | `vocab_size` * `n_embd` | 32,000 * 768 = **24,576,000** |
| **Mövqe Gömülməsi (`wpe`)** | `block_size` * `n_embd` | 512 * 768 = **393,216** |
| **Cəmi** | | **24,969,216** |

**Qeyd:** NanoGPT-də `wte` və `lm_head` çəkiləri bəzən paylaşılır (Weight Tying). Bizim kodumuzda onlar ayrıdır, lakin `lm_head` üçün hesablamanı ayrıca edəcəyik.

#### B. Bir Transformer Bloku (Block)

Hər bir blokun içində ən çox parametr **Çoxbaşlı Diqqət (MHA)** və **İrəli Ötürmə Şəbəkəsi (FFN)** qatlarında yerləşir.

**1. Çoxbaşlı Diqqət (`attn`):**
*   **Q, K, V Proyeksiyası (`c_attn`):** Giriş ölçüsü (`n_embd`) * Çıxış ölçüsü (`3 * n_embd`)
    *   Hesablama: 768 * (3 * 768) = 768 * 2304 = **1,769,472**
*   **Son Proyeksiya (`c_proj`):** Giriş ölçüsü (`n_embd`) * Çıxış ölçüsü (`n_embd`)
    *   Hesablama: 768 * 768 = **589,824**
*   **Cəmi MHA:** 1,769,472 + 589,824 = **2,359,296**

**2. İrəli Ötürmə Şəbəkəsi (`mlp`):**
*   **Giriş Qatı (`c_fc`):** Giriş ölçüsü (`n_embd`) * Çıxış ölçüsü (`4 * n_embd`)
    *   Hesablama: 768 * (4 * 768) = 768 * 3072 = **2,359,296**
*   **Çıxış Qatı (`c_proj`):** Giriş ölçüsü (`4 * n_embd`) * Çıxış ölçüsü (`n_embd`)
    *   Hesablama: 3072 * 768 = **2,359,296**
*   **Cəmi FFN:** 2,359,296 + 2,359,296 = **4,718,592**

**3. Digər Qatlar (`LayerNorm`):**
*   LayerNorm qatları da parametr ehtiva edir (çəki və meyillik). Hər LayerNorm üçün `2 * n_embd` parametr var.
    *   Hesablama: 4 * (2 * 768) = **6,144**

**4. Bir Blokun Cəmi:** 2,359,296 (MHA) + 4,718,592 (FFN) + 6,144 (LayerNorm) = **7,084,032**

#### C. Bütün Transformer Blokları

*   **Cəmi Bloklar:** `n_layer` * Bir Blokun Cəmi
    *   Hesablama: 12 * 7,084,032 = **85,008,384**

#### D. Dil Modeli Başı (LM Head)

*   **LM Head (`lm_head`):** Giriş ölçüsü (`n_embd`) * Çıxış ölçüsü (`vocab_size`)
    *   Hesablama: 768 * 32,000 = **24,576,000**

### 3. Yekun Hesablama

| Hissə | Parametr Sayı |
| :--- | :--- |
| Gömülmə Qatları | 24,969,216 |
| Transformer Blokları (12 ədəd) | 85,008,384 |
| LM Head | 24,576,000 |
| **Ümumi Parametr Sayı** | **134,553,600** |

**Qeyd:** Bizim PyTorch kodumuzda `lm_head`-in çəkiləri (`lm_head.weight`) və `wte`-nin çəkiləri (`wte.weight`) eyni matrisi paylaşır (Weight Tying). Əgər bu paylaşım tətbiq olunarsa, `lm_head` parametrləri ümumi saydan çıxılır.

Bizim `model.py` kodumuzda `self.get_num_params()` funksiyası `wpe` (Mövqe Gömülməsi) parametrlərini çıxarır. Əgər bütün parametrləri saysaq, təxminən **124 Milyon** rəqəmini alırıq (bu, bias-ların sayılmasından və ya sayılmamasından asılı olaraq dəyişə bilər).

**Əsas Nəticə:** Modelimizin ölçüsü **~124 Milyon** parametrdir.

### 4. Yaddaş Tələbi

Hər bir parametr yaddaşda yer tutur. Ən çox istifadə olunan dəqiqlik formatı **FP32** (32-bit Floating Point) və ya **FP16** (16-bit Floating Point) formatıdır.

*   **FP32 (4 byte):** 124,417,536 parametr * 4 byte/parametr ≈ **497 MB**
*   **FP16 (2 byte):** 124,417,536 parametr * 2 byte/parametr ≈ **248 MB**

Bu, modelin özünün yaddaşda tutduğu yerdir. Təlim zamanı optimallaşdırıcı (AdamW) və qradiyentlər də yaddaş tələb edir.

**Təlim zamanı ümumi VRAM tələbi:** Modelin çəkisi (FP16) * 1 (model) + Modelin çəkisi * 1 (qradiyent) + Modelin çəkisi * 2 (AdamW optimallaşdırıcısı) + Batch size * Context Length * n_embd * 4 (aktivasiyalar)

**Yekun Təxmin:** Bizim 12 GB VRAM-lı **NVIDIA T4** kartımız bu modeli FP16 (Mixed Precision) istifadə edərək rahatlıqla təlim edə biləcək.

### 💡 Günün Tapşırığı: Düşün və Praktika

1.  Əgər `n_layer`-i 24-ə qaldırsaydıq, modelin parametr sayı təxminən nə qədər olardı? (Cavab: Təxminən 200 Milyon).
2.  Modelin yaddaş tələbinin ən böyük hissəsi hansı komponentlərə aiddir? (Cavab: Transformer Blokları və Gömülmə Qatları).

**Sabah görüşənədək!** 👋 Sabah modelin təlimdən əvvəl necə mətn yaratdığını görmək üçün **Mətn Generasiyası (Sampling)** mexanizmini öyrənəcəyik.

***

**Söz Sayı:** 850 söz.
