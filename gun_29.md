# Gün 29: Təlimin Sonlandırılması və Modelin Hazırlanması 🏁

## 29.1. Təlimi Nə Vaxt Sonlandırmalı?

LLM təlimi həftələr, hətta aylar çəkə bilər. Lakin bizim 100M parametrli modelimiz üçün təlimi sonlandırmaq qərarı aşağıdakı iki əsas amilə əsaslanmalıdır:

1.  **Validasiya Loss-unun Dəyişməsi:** Əgər Validasiya Loss-u ardıcıl olaraq bir neçə epoxa ərzində azalmağı dayandırırsa və ya artmağa başlayırsa (Overfitting), təlimi dayandırmaq lazımdır. Bu texnika **Early Stopping (Erkən Dayandırma)** adlanır.
2.  **Mətn Generasiyasının Keyfiyyəti:** Modelin yaratdığı mətnləri yoxlayın. Əgər mətnlər axıcı, məntiqli və Azərbaycan dilinin qrammatikasına uyğundursa, bu, modelin kifayət qədər öyrəndiyini göstərir.

**Unutmayın:** Təlimi həmişə ən yaxşı **Validasiya Loss-u** olan Checkpoint-də dayandırın.

## 29.2. Modelin Hazırlanması (Final Model Export)

Təlim başa çatdıqdan sonra, biz modelin çəkilərini **təkcə proqnozlaşdırma (inference)** üçün istifadə edilə biləcək formata çevirməliyik.

**Təlimdən Fərqli Olaraq:**

*   **Optimallaşdırıcı (Optimizer):** Artıq lazım deyil.
*   **Təlim Parametrləri:** Artıq lazım deyil.
*   **Modelin Özü:** Yalnız modelin arxitekturası və öyrənilmiş çəkiləri lazımdır.

**Final Modelin Saxlanması:**

```python
# 1. Ən yaxşı Checkpoint-i yükləmək
checkpoint = torch.load('best_model_weights.pt')

# 2. Yeni bir model obyekti yaratmaq
final_model = GPTModel()

# 3. Çəkiləri yükləmək
final_model.load_state_dict(checkpoint['model_state_dict'])

# 4. Modeli CPU-ya köçürmək (Əgər GPU-da idisə)
final_model.to('cpu')

# 5. Modeli proqnozlaşdırma rejiminə keçirmək
final_model.eval()

# 6. Yalnız modelin çəkilərini saxlamaq (daha kiçik fayl)
torch.save(final_model.state_dict(), 'az_llm_100m_final.pt')
print("Final model çəkiləri 'az_llm_100m_final.pt' faylına yazıldı.")
```

## 29.3. Modelin Test Edilməsi (Generation)

Modeli yadda saxlamazdan əvvəl, onun mətn yaratma qabiliyyətini yoxlamalıyıq.

**`generate.py`**

```python
import torch
# GPTModel və Tokenizer-i import edin

def generate_text(model, tokenizer, start_text, max_new_tokens=100):
    """Modelin mətn yaratma funksiyası."""
    
    # 1. Giriş mətnini token ID-lərinə çevirmək
    encoded = tokenizer.encode(start_text)
    idx = torch.tensor(encoded.ids, dtype=torch.long).unsqueeze(0) # (1, T)
    
    # 2. Mətn yaratmaq
    # Modelin özündəki generate funksiyasını istifadə edirik
    # Bu funksiya hər dəfə bir token proqnozlaşdırır və onu girişə əlavə edir.
    generated_ids = model.generate(idx, max_new_tokens=max_new_tokens)
    
    # 3. Token ID-lərini mətnə çevirmək
    generated_text = tokenizer.decode(generated_ids[0].tolist())
    
    return generated_text

# ... (Modeli yükləmək və generate funksiyasını çağırmaq) ...
```

**Gündəlik Tapşırıq:** Təlimi dayandırmaq üçün ən yaxşı Validasiya Loss-u olan Checkpoint-i seçin. Modelin çəkilərini `az_llm_100m_final.pt` faylına yadda saxlayın.
