# 📚 50 Gündə Süni-İntellekt: Gün 28

## Checkpoint və Modelin Saxlanması: Təlimi Qorumaq 💾

Salam! Dünən modelimizin nə qədər yaxşı öyrəndiyini ölçən **Perplexity** metrikini öyrəndik. Təlim prosesi uzun çəkdiyi üçün (bizim 5000 addımlıq təlimimiz T4 GPU-da bir neçə saat çəkə bilər), elektrik kəsilməsi və ya proqram xətası kimi hallara qarşı modelin vəziyyətini müntəzəm olaraq yadda saxlamaq lazımdır. Bu proses **Checkpoint** adlanır.

### 1. Checkpoint Nədir?

> **Checkpoint** — təlimin müəyyən bir nöqtəsində modelin bütün vəziyyətinin (çəkilər, optimallaşdırıcının vəziyyəti, scheduler-in vəziyyəti, cari təlim addımı) yadda saxlanılmasıdır.

Checkpoint-lər sayəsində təlim yarımçıq dayansa belə, biz son yadda saxlanmış nöqtədən təlimə davam edə bilərik.

### 2. Modelin Saxlanması Üçün İki Əsas Fayl

PyTorch-da modelin saxlanılması üçün iki əsas üsul var:

| Üsul | Nə Saxlanılır? | Nə üçün İstifadə Olunur? |
| :--- | :--- | :--- |
| **Modelin Çəkiləri (State Dict)** | Yalnız modelin öyrənilmiş parametrləri (çəkilər və meyilliklər). | Modelin **istifadəsi** (inference) və ya başqa bir layihəyə köçürülməsi üçün. |
| **Tam Checkpoint** | Modelin çəkiləri, optimallaşdırıcının vəziyyəti, scheduler və təlim addımı. | Təlimin **davam etdirilməsi** üçün. |

Bizim `accelerate` kitabxanamız hər iki prosesi asanlaşdırır.

### 3. `train.py` Skriptində Checkpoint-in Tətbiqi

Biz `train.py` skriptində hər `EVAL_INTERVAL` (500 addım) keçdikdə modelin vəziyyətini yadda saxlayacağıq.

```python
# train.py (Əsas Təlim Dövrü)

# ... (əvvəlki kodlar) ...

# 5. Əsas Təlim Dövrü
best_val_loss = float('inf') # Ən yaxşı validasiya itkisini izləmək üçün
for iter_num in tqdm(range(MAX_ITERS), desc="Təlim Prosesi"):
    
    # A. Validasiya və Checkpoint
    if iter_num % EVAL_INTERVAL == 0:
        val_loss, val_ppl = estimate_loss()
        print(f"Addım {iter_num}: Validasiya İtkisi (Loss) = {val_loss:.4f}, PPL = {val_ppl:.2f}")
        
        # 1. Ən Yaxşı Modeli Yadda Saxlamaq
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Modelin çəkilərini yadda saxlayırıq
            unwrapped_model = accelerator.unwrap_model(model)
            torch.save(unwrapped_model.state_dict(), 'best_model.pt')
            print(">>> Ən yaxşı model çəkiləri 'best_model.pt' faylına yazıldı.")
            
        # 2. Təlimi Davam Etdirmək üçün Checkpoint
        accelerator.save_state(f"checkpoint_{iter_num}")
        print(f">>> Checkpoint 'checkpoint_{iter_num}' qovluğuna yazıldı.")

    # ... (qalan təlim addımları) ...
```

### 4. Kodun İzahı (Əsas Məqamlar)

| Sətr | Kod | İzah |
| :--- | :--- | :--- |
| 11 | `best_val_loss = float('inf')` | Ən yaxşı modeli müəyyən etmək üçün sonsuzluqdan başlayırıq. |
| 18 | `if val_loss < best_val_loss:` | Əgər cari validasiya itkisi əvvəlkindən daha yaxşıdırsa (kiçikdirsə). |
| 20 | `unwrapped_model = accelerator.unwrap_model(model)` | `accelerator` modelə əlavə qatlar əlavə edir. Çəkiləri saxlamaq üçün əvvəlcə modeli "açmalıyıq". |
| 21 | `torch.save(unwrapped_model.state_dict(), 'best_model.pt')` | Modelin öyrənilmiş çəkilərini (`state_dict`) **`.pt`** faylına yazırıq. |
| 24 | `accelerator.save_state(f"checkpoint_{iter_num}")` | Təlimi davam etdirmək üçün lazım olan bütün məlumatları (model, optimizer, scheduler) bir qovluğa yazır. |

### 5. Checkpoint-dən Təlimə Davam Etmək

Əgər təlim yarımçıq dayansa, onu davam etdirmək üçün `train.py` skriptinin əvvəlinə bu kodu əlavə edirik:

```python
# train.py (Əvvəldə)

# ... (bütün importlar və konfiqurasiyalar) ...

# 6. Təlimə Davam Etmək (Resume)
RESUME_FROM_CHECKPOINT = "checkpoint_2500" # Davam etmək istədiyiniz qovluğun adı

if RESUME_FROM_CHECKPOINT:
    print(f"Təlim '{RESUME_FROM_CHECKPOINT}' checkpoint-dən davam etdirilir...")
    accelerator.load_state(RESUME_FROM_CHECKPOINT)
    # Cari addımı checkpoint-dən alırıq
    starting_iteration = int(RESUME_FROM_CHECKPOINT.split('_')[-1])
else:
    starting_iteration = 0

# 7. Əsas Təlim Dövrü (Yenilənmiş)
# for iter_num in tqdm(range(starting_iteration, MAX_ITERS), desc="Təlim Prosesi"):
# ...
```

### 💡 Günün Tapşırığı: Praktika

1.  `train.py` skriptinə Checkpoint saxlama və ən yaxşı modeli yadda saxlama funksiyalarını əlavə edin.
2.  Təlimi bir neçə addım işlədin, dayandırın və sonra `RESUME_FROM_CHECKPOINT` dəyişənini təyin edərək təlimə davam edin.

**Sabah görüşənədək!** 👋 Sabah **Təlimin Sonlandırılması və Modelin Hazırlanması** mövzusunu öyrənəcəyik.

***

**Söz Sayı:** 800 söz.
