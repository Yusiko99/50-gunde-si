# Gün 37: Modelin Qiymətləndirilməsi və Nəticələrin Təhlili 📈

## 37.1. LLM Qiymətləndirmə Metodologiyası

Modelin təlimi başa çatdıqdan sonra, onun performansını obyektiv şəkildə ölçmək vacibdir. Qiymətləndirmə iki əsas metrika növünə əsaslanır:

### A. Intrinsik Metrikalar (Daxili)

Bunlar modelin dilin strukturunu nə qədər yaxşı öyrəndiyini ölçür.

1.  **Loss (İtki):** Təlim zamanı modelin proqnozlaşdırma səhvi.
2.  **Perplexity (PPL):** Modelin növbəti tokeni proqnozlaşdırmaqda nə qədər əmin olduğunu göstərən əsas metrika ($PPL = e^{Loss}$). **Aşağı PPL daha yaxşı model deməkdir.**

### B. Ekstrinsik Metrikalar (Xarici)

Bunlar modelin real dünya tapşırıqlarında (məsələn, sual-cavab, xülasələşdirmə) nə qədər faydalı olduğunu ölçür.

1.  **İnsan Qiymətləndirməsi:** Modelin yaratdığı mətnin axıcılıq, məntiqi ardıcıllıq və məlumatın dəqiqliyi baxımından insanlar tərəfindən qiymətləndirilməsi.
2.  **Benchmarklar:** Dilə xas olan standart test dəstləri (məsələn, Azərbaycan dilində sual-cavab testləri) üzərində modelin sınaqdan keçirilməsi.

## 37.2. 134M Parametrli Model üçün Gözləntilər

Modelin ölçüsü (134M) və təlim korpusunun həcmi (təxminən 1GB) nəzərə alınaraq, aşağıdakı nəticələr gözlənilir:

| Nəticə Parametri | Gözlənti | Məntiqi Əsas |
| :--- | :--- | :--- |
| **Axıcılıq** | Yüksək | Model Azərbaycan dilinin qrammatik və sintaktik qaydalarını öyrənmək üçün kifayət qədər məlumat görüb. |
| **Məntiqi Ardıcıllıq** | Orta | Kiçik model olduğu üçün uzun və mürəkkəb məntiqi əlaqələri qorumaqda çətinlik çəkə bilər. |
| **Bilik Dərinliyi** | Səthi | Modelin biliyi yalnız təlim korpusu ilə məhdudlaşır. Xüsusi və ya aktual məlumatlar haqqında bilikləri məhdud olacaq. |
| **Halüsinasiya** | Orta Risk | Model bilmədiyi suallara məntiqli görünən, lakin faktiki səhv olan cavablar (halüsinasiya) yarada bilər. |

## 37.3. Nəticələrin Təhlili

Qiymətləndirmə nəticələri modelin təkmilləşdirilməsi üçün yol xəritəsini müəyyənləşdirir:

1.  **Yüksək PPL:** Korpusun keyfiyyəti və ya həcmi qeyri-kafi ola bilər. Daha çox və daha təmiz məlumat toplanmalıdır.
2.  **Yaxşı PPL, Lakin Zəif Məntiq:** Modelin ölçüsü (n_layer, n_embd) tapşırıq üçün çox kiçik ola bilər. Resurslar imkan verərsə, modelin ölçüsü artırılmalıdır.
3.  **Overfitting:** Validasiya Loss-u artırsa, təlim dayandırılmalı və **Dropout** dərəcəsi artırılmalıdır.

**Nəticə:** Modelin qiymətləndirilməsi, təlim prosesinin elmi əsasını təşkil edir və növbəti iterasiyalar üçün obyektiv qərar qəbul etməyə imkan verir.
