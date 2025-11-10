# Gün 40: Etik Mülahizələr və Məsuliyyətli Süni İntellekt ⚖️

## 40.1. Məsuliyyətli Süni İntellekt

Siz artıq güclü bir alət – Böyük Dil Modeli yaratmısınız. Bu güc böyük məsuliyyət tələb edir. **Məsuliyyətli Süni İntellekt (Responsible AI)** modelinizin cəmiyyətə zərər vermədən, ədalətli və şəffaf şəkildə istifadə edilməsini təmin edir.

## 40.2. Etik Mülahizələr

### A. Məlumatın Qərəzliliyi (Bias)

Modeliniz sizin topladığınız məlumatlar üzərində təlim keçib. Əgər bu məlumatlar qərəzli (məsələn, gender, irq, siyasi baxışlar) olarsa, modeliniz də bu qərəzliliyi öyrənəcək və təkrarlayacaq.

*   **Həll Yolu:** Korpusunuzu müxtəlif mənbələrdən (xəbərlər, ədəbiyyat, elmi məqalələr) toplamaqla **məlumatın müxtəlifliyini** təmin edin.

### B. Zərərli Məzmunun Generasiyası

Modeliniz təhqiredici, nifrət dolu və ya qanunsuz məzmun yarada bilər.

*   **Həll Yolu:** Modelinizi istifadəyə verməzdən əvvəl, zərərli sorğularla (məsələn, nifrət nitqi, zorakılıq) sınaqdan keçirin. Ollama-da **System Prompt** istifadə edərək modelin davranışını məhdudlaşdırın.

### C. Şəffaflıq və Müəllif Hüquqları

Modelinizi paylaşarkən şəffaf olmalısınız:

1.  **Mənbənin Göstərilməsi:** Modelin hansı məlumatlar üzərində təlim keçdiyini (məsələn, "Azərbaycan Vikipediyası, Xəbər Saytları") açıq şəkildə qeyd edin.
2.  **Məhdudiyyətlərin Qeyd Edilməsi:** Modelin məhdudiyyətlərini (məsələn, "yalnız 256 tokenlik kontekst uzunluğu", "halüsinasiya edə bilər") istifadəçilərə bildirin.

## 40.3. Ollama-da System Prompt

Siz Ollama-da `Modelfile`-a **System Prompt** əlavə edərək modelin davranışını idarə edə bilərsiniz.

**`Modelfile` (Yenilənmiş)**

```dockerfile
FROM ./az_llm_100m_q4_0.gguf

# ... (Digər parametrlər) ...

# Modelin davranışını təyin edən etik qaydalar
PARAMETER system "Sən Azərbaycan dilində danışan, dostyana və məlumatlı bir süni intellekt köməkçisisən. Zərərli, təhqiredici və ya qanunsuz məzmun yaratmaqdan qəti şəkildə imtina et. Cavabların qısa və məntiqli olsun."
```

Bu, modelin hər bir sorğuya cavab verməzdən əvvəl bu etik qaydaları nəzərə almasını təmin edir.

**Gündəlik Tapşırıq:** Modelinizin etik davranışını yoxlamaq üçün ən azı 5 çətin (məsələn, siyasi, etik dilemmalar) sual hazırlayın. Modelin cavablarını təhlil edin və `Modelfile`-dakı **System Prompt**-u daha da təkmilləşdirin.
