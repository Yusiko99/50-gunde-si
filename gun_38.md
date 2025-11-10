# Gün 38: Modelin Təkmilləşdirilməsi (Iterasiya) 🔄

## 38.1. Təkmilləşdirmə Dövrü

Süni İntellekt layihələri heç vaxt bitmir, onlar sadəcə təkmilləşir. Modelinizin nəticələrini təhlil etdikdən sonra, onu daha yaxşı etmək üçün **Iterasiya Dövrünə** başlamalısınız.

**Iterasiya Dövrü:**

1.  **Analiz:** Modelin zəif tərəflərini müəyyənləşdirin (məsələn, "siyasi mövzularda zəifdir", "qısa cümlələr qurur").
2.  **Hipotez:** Zəifliyin səbəbini güman edin (məsələn, "korpusda siyasi mətnlər azdır").
3.  **Eksperiment:** Hipotezi yoxlamaq üçün dəyişiklik edin (məsələn, "daha çox siyasi xəbər saytından məlumat topla").
4.  **Təlim:** Modeli yenidən təlim edin.
5.  **Qiymətləndirmə:** Nəticələri müqayisə edin.

## 38.2. Məlumatın Təkmilləşdirilməsi

Modelin keyfiyyətini artırmağın ən təsirli yolu **məlumatın keyfiyyətini** artırmaqdır.

| Problem | Həll Yolu |
| :--- | :--- |
| **Dilin Çirklənməsi** | Təmizləmə skriptinə (Gün 8) daha sərt qaydalar əlavə edin (məsələn, 5%-dən çox ingilis sözü olan sətirləri silmək). |
| **Mövzu Çatışmazlığı** | Yeni, spesifik mənbələr (məsələn, tibb, texnologiya forumları) əlavə edin. |
| **Təkrarlanan Mətn** | Təkrarlanan sətirləri silməklə yanaşı, oxşar sətirləri də silmək üçün **Simhash** kimi alqoritmlərdən istifadə edin. |

## 38.3. Modelin Təkmilləşdirilməsi

Modelin arxitekturasında kiçik dəyişikliklər böyük fərq yarada bilər:

1.  **Kontekst Uzunluğunun Artırılması:** `block_size`-ı 256-dan 512-yə artırın. Bu, modelin daha uzun cümlələri başa düşməsinə kömək edəcək. **Diqqət:** Bu, VRAM tələbini artıracaq.
2.  **Öyrənmə Sürətinin Tənzimlənməsi:** Təlimin sonunda öyrənmə sürətini azaltmaq (Learning Rate Decay) modelin daha dəqiq nəticələr verməsinə kömək edir.
3.  **Daha Yaxşı Tokenizator:** BPE əvəzinə **WordPiece** və ya **SentencePiece** kimi daha mürəkkəb tokenizatorları sınaqdan keçirin.

## 38.4. Günün Tapşırığı: Təkmilləşdirmə Planı

Modelinizin ən böyük zəifliyini müəyyənləşdirin və onu aradan qaldırmaq üçün **üç addımlıq** təkmilləşdirmə planı hazırlayın.

**Nümunə Plan:**

1.  **Analiz:** Modelin cavabları çox qısadır.
2.  **Hipotez:** Kontekst uzunluğu (256) qısa cümlələrə öyrəşməsinə səbəb olur.
3.  **Eksperiment:** `block_size`-ı 512-yə artır və təlimi yenidən başlat.

Bu planı sənədləşdirin.
