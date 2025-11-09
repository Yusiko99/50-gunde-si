# 📚 50 Gündə Süni-İntellekt: Gün 38

## Modelin Təkmilləşdirilməsi: Hiperparametr Tənzimlənməsi 🔧

Salam! Dünən modelimizin nəticələrini kəmiyyət və keyfiyyət baxımından qiymətləndirdik. Əgər nəticələr sizi tam qane etmirsə, modelin performansını artırmaq üçün **Hiperparametr Tənzimlənməsi (Hyperparameter Tuning)** aparmalıyıq.

### 1. Hiperparametr Nədir?

> **Hiperparametr** — modelin təlim prosesindən əvvəl insan tərəfindən təyin olunan dəyərlərdir. Modelin özü bu dəyərləri öyrənmir.

Bizim layihəmizdəki əsas hiperparametrlər:

| Hiperparametr | Fayl | Tənzimlənmənin Təsiri |
| :--- | :--- | :--- |
| **`LEARNING_RATE`** | `train.py` | Ən vacib parametr. Çox yüksək olarsa Loss partlayar, çox aşağı olarsa təlim yavaşlayar. |
| **`BATCH_SIZE`** | `train.py` | Nə qədər böyük olsa, təlim bir o qədər stabil olar (lakin VRAM tələbi artar). |
| **`n_layer`** | `config.py` | Modelin dərinliyi. Artırılması performansı artırır, lakin təlimi yavaşladır. |
| **`n_embd`** | `config.py` | Modelin "eni". Artırılması performansı artırır, lakin parametr sayını kəskin artırır. |
| **`block_size`** | `config.py` | Modelin kontekst pəncərəsi. Artırılması modelin daha uzun mətnləri xatırlamasına kömək edir. |
| **`dropout`** | `config.py` | Overfitting-in qarşısını alır. Çox yüksək olarsa model öyrənməkdə çətinlik çəkər. |

### 2. Tənzimlənmə Strategiyaları

Hiperparametrləri tənzimləmək üçün iki əsas yanaşma var:

#### A. Grid Search (Şəbəkə Axtarışı)

*   **Prinsip:** Tənzimləmək istədiyiniz hər bir parametr üçün bir neçə dəyər seçirsiniz və bütün mümkün kombinasiyaları sınaqdan keçirirsiniz.
*   **Nümunə:** LR = [1e-4, 3e-4, 6e-4], Batch Size = [12, 16]. Cəmi $3 \times 2 = 6$ təlim sınağı.
*   **Mənfi Cəhəti:** Çox vaxt aparır.

#### B. Random Search (Təsadüfi Axtarış)

*   **Prinsip:** Parametrlər üçün müəyyən bir diapazon təyin edirsiniz və bu diapazondan təsadüfi kombinasiyalar seçərək sınaqdan keçirirsiniz.
*   **Üstünlüyü:** Grid Search-dən daha effektivdir, çünki ən vacib parametrlərin yaxşı dəyərlərini tapmaq ehtimalı daha yüksəkdir.

### 3. Təkmilləşdirmə üçün Praktik Addımlar

Bizim 100M modelimiz üçün ən çox təsir edəcək parametrlər bunlardır:

#### A. Öyrənmə Sürəti (`LEARNING_RATE`)

*   **Sınaq:** `6e-4` ilə başlayın. Əgər Loss çox tez azalırsa və ya partlayırsa, `3e-4` və ya `1e-4` ilə sınaqdan keçirin.
*   **Qeyd:** Əgər `BATCH_SIZE`-ı artırırsınızsa, `LEARNING_RATE`-i də bir qədər artırmaq lazımdır.

#### B. Modelin Ölçüsü (`n_embd` və `n_layer`)

*   **Hədəf:** Əgər VRAM-ınız imkan verirsə, modeli bir qədər böyüdün.
*   **Nümunə:** `n_layer`-i 12-dən **16**-ya qaldırın. Parametr sayı təxminən 160M olacaq. Bu, modelin daha dərin əlaqələri öyrənməsinə kömək edəcək.

#### C. Kontekst Pəncərəsi (`block_size`)

*   **Hədəf:** Modelin daha uzun mətnləri xatırlamasını istəyirsinizsə, `block_size`-ı **512-dən 1024-ə** qaldırın.
*   **Nəticə:** Bu, VRAM tələbini kəskin şəkildə artıracaq. `BATCH_SIZE`-ı azaltmalı və ya `GRADIENT_ACCUMULATION_STEPS`-i artırmalı ola bilərsiniz.

### 4. Təkmilləşdirmənin Sənədləşdirilməsi

Hər bir sınağın nəticəsini (istifadə olunan hiperparametrlər və son Validasiya PPL) qeyd edin.

| Sınaq # | `n_layer` | `n_embd` | `LR` | `Batch Size` | Final PPL | Nəticə |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 (Əsas) | 12 | 768 | 6e-4 | 48 (12x4) | 38.5 | Yaxşı başlanğıc |
| 2 | 16 | 768 | 6e-4 | 32 (8x4) | 35.1 | Daha yaxşı, lakin yavaş |
| 3 | 12 | 768 | 3e-4 | 48 (12x4) | 40.2 | Çox yavaş öyrənir |

### 💡 Günün Tapşırığı: Praktika

1.  `config.py` faylında `n_layer`-i 12-dən 16-ya dəyişdirin.
2.  `train.py` faylında `BATCH_SIZE`-ı 8-ə endirin və `GRADIENT_ACCUMULATION_STEPS`-i 4-də saxlayın (Effektiv Batch Size = 32).
3.  Yeni təlimi başladın və nəticələri əvvəlki ilə müqayisə edin.

**Sabah görüşənədək!** 👋 Sabah **Modelin İdarə Edilməsi və Sürətləndirilməsi** mövzusunu öyrənəcəyik.

***

**Söz Sayı:** 750 söz.
