# Gün 40: Etik Mülahizələr və Məsuliyyətli Süni İntellekt ⚖️

## 40.1. Məsuliyyətli Süni İntellekt (Responsible AI)

Böyük Dil Modellərinin inkişafı, etik və sosial məsuliyyətləri də özü ilə gətirir. **Məsuliyyətli Süni İntellekt** çərçivəsi, modelin ədalətli, şəffaf və cəmiyyətə zərər verməyən şəkildə istifadə edilməsini təmin edir.

## 40.2. Əsas Etik Risk Məqamları

### A. Məlumatın Qərəzliliyi (Bias)

Modelin təlim keçdiyi korpusda mövcud olan sosial qərəzliliklər (məsələn, gender, etnik mənsubiyyət, siyasi baxışlar) model tərəfindən öyrənilir və təkrarlanır.

*   **Məntiq:** Model korpusdakı statistik əlaqələri öyrənir. Əgər korpusda müəyyən bir qrup haqqında mənfi assosiasiyalar daha çoxdursa, model də bu assosiasiyaları gücləndirəcək.
*   **Həll Yolu:** Korpusun müxtəlifliyini artırmaq və təlimdən əvvəl qərəzli mətnləri aşkar edib təmizləmək.

### B. Zərərli Məzmunun Generasiyası

Modelin nifrət nitqi, təhqiredici və ya qanunsuz məzmun yaratma riski mövcuddur.

*   **Həll Yolu:** **Guardrails (Mühafizə Mexanizmləri)** tətbiq etmək. Ollama-da **System Prompt** vasitəsilə modelin davranışını məhdudlaşdırmaq ən sadə yoldur.

## 40.3. Ollama-da System Prompt ilə Davranışın İdarə Edilməsi

**System Prompt** modelin hər bir sorğuya cavab verməzdən əvvəl nəzərə almalı olduğu əsas qaydalar toplusudur.

**`Modelfile` (Etik Qaydalarla Yenilənmiş)**

```dockerfile
FROM ./az_llm_100m_q4_0.gguf

# ... (Digər parametrlər) ...

# Modelin etik davranışını təyin edən sistem təlimatı
PARAMETER system "Sən Azərbaycan dilində danışan, dostyana və məlumatlı bir süni intellekt köməkçisisən. Zərərli, təhqiredici, diskriminasiya edən və ya qanunsuz məzmun yaratmaqdan qəti şəkildə imtina et. Cavabların faktiki məlumatlara əsaslansın."
```

**Məntiq:** Bu təlimat, modelin daxili neyron şəbəkəsinə etik məhdudiyyətlər tətbiq edir və modelin çıxışını arzuolunan çərçivədə saxlayır.

## 40.4. Şəffaflıq və Sənədləşdirmə

Modelin etik istifadəsi üçün şəffaflıq vacibdir:

1.  **Mənbənin Açıqlanması:** Modelin hansı məlumatlar üzərində təlim keçdiyini (korpusun mənbələri) açıq şəkildə sənədləşdirmək.
2.  **Məhdudiyyətlərin Qeyd Edilməsi:** Modelin məhdudiyyətlərini (məsələn, halüsinasiya riski, bilik kəsilmə tarixi) istifadəçilərə bildirmək.

**Nəticə:** LLM tərtibatçısı olaraq, modelin texniki funksionallığı qədər, onun etik və sosial təsirlərinə görə də məsuliyyət daşıyırıq.
