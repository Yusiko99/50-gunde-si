# Gün 29: Təlimin Sonlandırılması və Modelin Hazırlanması 🏁

## 29.1. Təlimin Sonlandırılması (Early Stopping)

LLM təlimi üçün optimal sonlandırma nöqtəsi, modelin **ən yaxşı ümumiləşdirmə qabiliyyətinə** malik olduğu nöqtədir. Bu, adətən **Validasiya Loss-unun minimuma çatdığı** nöqtədir.

**Məntiq:** Təlim Loss-u azalmağa davam etsə də, Validasiya Loss-u artmağa başlayırsa (Overfitting), təlimi dərhal dayandırmaq lazımdır. Bu texnika **Early Stopping (Erkən Dayandırma)** adlanır.

**Erkən Dayandırma Kriteriyası:**

1.  Validasiya Loss-u ardıcıl olaraq `Patience` (məsələn, 3) epoxa ərzində yaxşılaşmırsa.
2.  Təlim ən yaxşı Validasiya Loss-u olan Checkpoint-də dayandırılır.

## 29.2. Modelin Hazırlanması (Inference Export)

Təlim başa çatdıqdan sonra, modelin çəkiləri **proqnozlaşdırma (Inference)** üçün optimallaşdırılmış formata çevrilməlidir.

**Təlim Vəziyyətindən Fərqlər:**

*   **Optimallaşdırıcı:** Təlim üçün lazım olan optimallaşdırıcı vəziyyəti (məsələn, AdamW-nin momentləri) silinir.
*   **Model Rejimi:** Model `model.eval()` rejiminə keçirilir.

**Praktika: Final Modelin Saxlanması**

```python
import torch
import os
# GPTModel sinfini import edin

# 1. Ən yaxşı Checkpoint-i yükləmək
CHECKPOINT_DIR = "checkpoints/best_model"
if not os.path.exists(CHECKPOINT_DIR):
    print("Xəta: Ən yaxşı Checkpoint tapılmadı.")
    exit()

# 2. Modelin vəziyyətini yükləmək (accelerate-dən)
# accelerate load_state funksiyası modelin çəkilərini yükləyir.
model = GPTModel(vocab_size=32000, block_size=256, n_layer=12, n_head=12, n_embd=768)
# Bu hissəni accelerate olmadan icra etmək üçün:
# model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, 'pytorch_model.bin')))

# 3. Modeli proqnozlaşdırma rejiminə keçirmək
model.eval()

# 4. Yalnız modelin çəkilərini saxlamaq (ən yüngül format)
torch.save(model.state_dict(), 'az_llm_100m_final.pt')
print("Final model çəkiləri 'az_llm_100m_final.pt' faylına yazıldı.")
```

## 29.3. Modelin Test Edilməsi (Generasiya)

Modelin proqnozlaşdırma rejimində düzgün işlədiyini yoxlamaq üçün **Generasiya (Mətn Yaratma)** testi aparılır.

**Məntiq:** Generasiya zamanı modelin çəkiləri dəyişmir. Model yalnız verilmiş giriş ardıcıllığına əsasən növbəti tokenin ehtimalını hesablayır.

**Generasiya Addımları:**

1.  Giriş mətni tokenizasiya edilir.
2.  Token ID-ləri modelə verilir.
3.  Model növbəti tokenin ehtimalını (Logits) qaytarır.
4.  Bu ehtimallardan **Sampling** (Gün 20-də öyrənilən) vasitəsilə bir token seçilir.
5.  Seçilmiş token giriş ardıcıllığına əlavə edilir və proses təkrarlanır.

**Qeyd:** Bu mərhələdə modelin çəkiləri `az_llm_100m_final.pt` faylında saxlanılır. Bu fayl növbəti mərhələdə **Hugging Face** formatına çevriləcək.
