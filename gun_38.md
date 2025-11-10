# Gün 38: Modelin Təkmilləşdirilməsi (Iterasiya) 🔄

## 38.1. Təkmilləşdirmə Dövrü (Iterative Improvement)

Süni İntellekt layihələri statik deyil, dinamikdir. Modelin ilkin qiymətləndirilməsindən sonra, onun performansını artırmaq üçün **Iterasiya Dövrü** başlayır.

**Məntiq:** Hər bir iterasiya, modelin zəif tərəflərini aradan qaldırmaq üçün məlumat, arxitektura və ya təlim parametrlərində dəyişikliklər etməkdən ibarətdir.

**Iterasiya Dövrünün Mərhələləri:**

1.  **Analiz:** Modelin zəifliklərini (məsələn, qısa cavablar, qərəzli məlumat) müəyyənləşdirmək.
2.  **Hipotez:** Zəifliyin səbəbini (məsələn, qeyri-kafi korpus, kiçik kontekst pəncərəsi) təyin etmək.
3.  **Eksperiment:** Hipotezi yoxlamaq üçün dəyişiklik etmək.
4.  **Qiymətləndirmə:** Yeni modelin nəticələrini əvvəlki ilə müqayisə etmək.

## 38.2. Məlumatın Təkmilləşdirilməsi

Modelin keyfiyyətini artırmağın ən təsirli yolu **təlim məlumatının keyfiyyətini və müxtəlifliyini** artırmaqdır.

| Problem | Həll Yolu | Məntiqi Əsas |
| :--- | :--- | :--- |
| **Məhdud Mövzu Bilikləri** | Yeni, spesifik mənbələr (məsələn, elmi jurnallar, texnoloji bloqlar) əlavə etmək. | Modelin bilik bazasını genişləndirmək. |
| **Təkrarlanan Məlumat** | Təmizləmə skriptinə **Simhash** kimi alqoritmləri əlavə etmək. | Modelin eyni məlumatı dəfələrlə görməsinin qarşısını almaq. |
| **Dilin Çirklənməsi** | Təmizləmə prosesində (Gün 8) xarici dildə olan mətnlərin faizini yoxlamaq və yüksək faizli sətirləri silmək. | Modelin yalnız Azərbaycan dilinə fokuslanmasını təmin etmək. |

## 38.3. Modelin Təkmilləşdirilməsi

Modelin arxitekturasında və təlim parametrlərində dəyişikliklər:

1.  **Kontekst Pəncərəsinin Artırılması:** `block_size`-ı 256-dan 512-yə artırmaq. **Məntiq:** Modelin daha uzun cümlələr və paraqraflar arasındakı əlaqələri başa düşməsinə imkan verir. **Diqqət:** Bu, VRAM tələbini artıracaq.
2.  **Öyrənmə Sürətinin Tənzimlənməsi:** Təlimin sonunda öyrənmə sürətini azaltmaq (Learning Rate Decay) modelin daha dəqiq nəticələr verməsinə kömək edir.
3.  **Daha Böyük Model:** Resurslar imkan verərsə, `n_layer` (qat sayı) və ya `n_embd` (embedding ölçüsü) artırmaqla modelin mürəkkəbliyini artırmaq.

**Nəticə:** Təkmilləşdirmə prosesi, elmi metodologiyaya əsaslanan, daimi sınaq və nəticələrin təhlili tələb edən bir dövrdür.
