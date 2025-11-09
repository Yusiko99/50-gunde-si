# 📚 50 Gündə Süni-İntellekt: Gün 37

## Modelin Qiymətləndirilməsi və Nəticələrin Təhlili 📊

Salam! Dünən layihəmizi GitHub-da paylaşmağı öyrəndik. Bu gün isə təlimin nəticələrini obyektiv şəkildə qiymətləndirməyə və modelimizin Azərbaycan dilini nə qədər yaxşı öyrəndiyini təhlil etməyə başlayırıq.

### 1. Kəmiyyət Qiymətləndirməsi (Quantitative Evaluation)

Kəmiyyət qiymətləndirməsi rəqəmlərə əsaslanır. Bizim əsas kəmiyyət metrikimiz **Perplexity (PPL)**-dir.

#### A. PPL-in Təhlili

Təlim bitdikdən sonra əldə etdiyimiz son **Validasiya PPL** dəyəri modelimizin nə qədər yaxşı olduğunu göstərir.

| PPL Dəyəri | İzah |
| :--- | :--- |
| **> 100** | Model demək olar ki, heç nə öyrənməyib. (Təlimsiz modelin PPL-i $\approx 1280$ idi). |
| **50 - 100** | Model təməl qrammatik qaydaları öyrənib, lakin mənalı mətn yaratmaqda çətinlik çəkir. |
| **10 - 50** | **Yaxşı nəticə.** Model səlis və mənalı mətnlər yarada bilir. |
| **< 10** | **Əla nəticə.** Model dilin incəliklərini başa düşür. |

**Bizim Hədəfimiz:** 100M parametreli model və 100M tokenlik məlumatla **PPL-i 30-40 arasına** endirmək realistik bir hədəfdir.

#### B. Niyə PPL Təkbaşına Kifayət Deyil?

PPL modelin **səlisliyini** (fluency) ölçür, lakin **məntiqliliyini** (coherence) və **faydalılığını** (usefulness) ölçmür.

Məsələn, model çox aşağı PPL ilə qrammatik cəhətdən qüsursuz, lakin tamamilə mənasız bir mətn yarada bilər.

### 2. Keyfiyyət Qiymətləndirməsi (Qualitative Evaluation)

Keyfiyyət qiymətləndirməsi modelin yaratdığı mətnlərin insan tərəfindən oxunub qiymətləndirilməsidir.

#### A. Sınaq Promptları (Test Prompts)

Modelin müxtəlif qabiliyyətlərini yoxlamaq üçün xüsusi sınaq promptları hazırlayırıq:

| Qabiliyyət | Sınaq Promptu | Gözlənilən Cavab |
| :--- | :--- | :--- |
| **Fakt Bilikləri** | "Azərbaycanın ən hündür dağı hansıdır?" | "Bazardüzü dağıdır." |
| **Yaradıcılıq** | "Qədim Bakı haqqında bir hekayə yaz." | Qısa, maraqlı bir hekayə. |
| **Qrammatika** | "Mən dünən kitab oxu." (Səhv cümlə) | "Mən dünən kitab oxudum." (Düzəliş) |
| **Söhbət** | "Salam, necəsən?" | "Salam, mən bir süni intellekt modeliyəm. Sənə necə kömək edə bilərəm?" |

#### B. Keyfiyyət Təhlili Skripti

Bizim `load_model.py` skriptimizdəki `generate_text` funksiyasını istifadə edərək bu sınaqları avtomatlaşdıra bilərik.

```python
# evaluate_model.py
import load_model # Dünənki skripti daxil edirik

test_prompts = [
    "Azərbaycanın paytaxtı Bakı",
    "Mən dünən kitab oxu",
    "Süni intellekt nədir?",
    "Qarabağ haqqında bir cümlə yaz.",
]

print("--- Modelin Keyfiyyət Qiymətləndirilməsi ---")

for prompt in test_prompts:
    response = load_model.generate_text(prompt, max_new_tokens=50)
    
    print(f"\n[PROMPT]: {prompt}")
    print(f"[CAVAB]: {response}")
    print("-" * 20)
```

### 3. Nəticələrin Sənədləşdirilməsi

Təlimin nəticələrini sənədləşdirmək, layihənizin etibarlılığını artırır.

**GitHub README.md-yə əlavə edin:**
*   **Final Validasiya PPL:** XX.XX
*   **Model Ölçüsü:** 124 Milyon Parametr (Q4_K_M ilə 62 MB)
*   **Sınaq Nəticələri:** Yuxarıdakı sınaq promptlarının ən yaxşı cavablarını əlavə edin.

### 💡 Günün Tapşırığı: Praktika

1.  `evaluate_model.py` faylını yaradın və sınaq promptlarınıza əlavələr edin.
2.  Təlim olunmuş modelinizi (`best_model.pt`) yükləyərək skripti icra edin.
3.  Modelin cavablarını diqqətlə oxuyun və qeydlər aparın.

**Sabah görüşənədək!** 👋 Sabah **Modelin Təkmilləşdirilməsi: Hiperparametr Tənzimlənməsi** mövzusunu öyrənəcəyik.

***

**Söz Sayı:** 750 söz.
