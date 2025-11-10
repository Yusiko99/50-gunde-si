# Gün 27: Validasiya və Qiymətləndirmə 🔬

## 27.1. Validasiya Nədir?

**Validasiya (Validation)** təlim prosesinin ayrılmaz hissəsidir. Bizim modelimiz təlim məlumatları üzərində öyrənir, lakin biz onun **görmədiyi** məlumatlar üzərində nə qədər yaxşı işlədiyini bilməliyik. Bu məqsədlə, Gün 12-də məlumatımızın 10%-ni **Validasiya Dəsti** kimi ayırmışdıq.

**Validasiyanın Əsas Məqsədi:**

1.  **Overfitting-in Qarşısını Almaq:** Modelin əzbərləməyib, həqiqətən öyrəndiyini yoxlamaq.
2.  **Hiperparametrlərin Seçimi:** Ən yaxşı öyrənmə sürəti, Batch Size və s. kimi hiperparametrləri seçməyə kömək etmək.

## 27.2. Modelin Qiymətləndirilməsi

Təlim başa çatdıqdan sonra modelin performansını ölçmək üçün istifadə olunan əsas metrikalar bunlardır:

### A. Perplexity (PPL)

Gün 26-da öyrəndiyimiz kimi, PPL dil modelinin nə qədər yaxşı proqnozlaşdırdığını göstərir.

### B. Mətn Generasiyası (Text Generation)

LLM-in əsas məqsədi mətn yaratmaqdır. Buna görə də, modelin keyfiyyətini qiymətləndirməyin ən yaxşı yolu, onun yaratdığı mətnləri **insan gözü ilə** oxumaqdır.

**Qiymətləndirmə Kriteriyaları:**

1.  **Axıcılıq (Fluency):** Mətn qrammatik cəhətdən düzgündürmü?
2.  **Məntiqlilik (Coherence):** Mətn mövzu daxilində məntiqli və ardıcılmı?
3.  **Uyğunluq (Relevance):** Modelin cavabı verilən suala və ya başlanğıc mətndəki kontekstə uyğundurmu?

## 27.3. Praktika: Validasiya Loss-unun Hesablanması

Gün 26-da `estimate_loss` funksiyasını təqdim etdik. Bu funksiya validasiya dəsti üzərində modelin performansını ölçür.

**`estimate_loss` funksiyasının əsas addımları:**

1.  **`model.eval()`:** Modeli qiymətləndirmə rejiminə keçirir. Bu rejimdə **Dropout** və **Batch Normalization** (bizim modeldə yoxdur) kimi laylar deaktiv edilir.
2.  **`torch.no_grad()`:** Qradiyentlərin hesablanmasını dayandırır. Bu, VRAM-a qənaət edir və hesablama sürətini artırır.
3.  **Loss-un Hesablanması:** Validasiya dəstinin bütün Batch-ləri üzərində Loss hesablanır.
4.  **`model.train()`:** Qiymətləndirmə bitdikdən sonra model təlim rejiminə qaytarılır.

## 27.4. Modelin Saxlanması (Checkpointing)

Validasiya Loss-u ən aşağı olan modeli saxlamaq çox vacibdir.

**Ən Yaxşı Modelin Saxlanması:**

```python
# Tutaq ki, bu, ən yaxşı validasiya loss-udur
best_val_loss = float('inf') 

# Təlim dövrü daxilində, hər 1000 addımda:
if val_loss < best_val_loss:
    best_val_loss = val_loss
    
    # Modelin çəkilərini yadda saxlamaq
    torch.save(model.state_dict(), 'best_model_weights.pt')
    print("Yeni ən yaxşı model çəkiləri yadda saxlanıldı!")
```

**Gündəlik Tapşırıq:** `train_accelerate.py` skriptinizdə ən yaxşı validasiya loss-una əsasən model çəkilərini yadda saxlama mexanizmini tətbiq edin.
