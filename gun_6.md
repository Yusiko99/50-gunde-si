# Gün 6: Dataset İnşası I: Məlumat Mənbələrinin Təyini 🗺️

## 6.1. Niyə Öz Korpusumuzu Qururuq?

Əvvəlki günlərdə qeyd etdiyimiz kimi, biz hazır **azcorpus** kimi korpuslardan istifadə etməyəcəyik. Bunun əsas səbəbi, LLM-lərin necə işlədiyini **tamamilə sıfırdan** öyrənməkdir. Korpus LLM-in qidasıdır. Qidanı özümüz hazırlayaraq, onun tərkibini və keyfiyyətini tam nəzarətdə saxlayırıq.

**Korpus (Corpus)** – təlim üçün istifadə olunan böyük həcmli mətn toplusudur. Bizim LLM-imiz Azərbaycan dilini bu korpusdan öyrənəcək.

## 6.2. Azərbaycan Dili Mənbələrinin Təyini

Azərbaycan dili üçün böyük və keyfiyyətli mətn mənbələri tapmaq ingilis dili qədər asan deyil. Bizim məqsədimiz **müxtəlif mövzuları** əhatə edən, **yüksək keyfiyyətli** və **açıq şəkildə əlçatan** mənbələr tapmaqdır.

Bizim korpusumuz üçün potensial mənbələr:

| Mənbə Növü | Nümunə Mənbələr | Niyə Vacibdir? |
| :--- | :--- | :--- |
| **Vikipediya** | Azərbaycan Vikipediyası | **Elmi, tarixi və ensiklopedik** məlumatlar verir. Dilin rəsmi və neytral tonunu öyrədir. |
| **Xəbər Saytları** | Azertac, Report, Qafqazinfo və s. | **Aktual hadisələr, siyasi və iqtisadi** terminologiyanı öyrədir. |
| **Rəsmi Sənədlər** | Qanunvericilik bazası, Nazirlik saytları | **Hüquqi və rəsmi** dilin strukturunu öyrədir. |
| **Ədəbiyyat** | Açıq mənbəli elektron kitabxanalar | **Bədii, emosional və zəngin** dil quruluşunu öyrədir. |
| **Forumlar/Bloqlar** | Texnoloji, sosial forumlar | **Danışıq dilini, jarqonları** və qeyri-rəsmi üslubu öyrədir. |

**Diqqət:** Biz bu mənbələrdən məlumatları **Web Scraping** (Vebdən Məlumat Çəkmə) üsulu ilə toplayacağıq. Bu, etik və hüquqi məsələlərə diqqət yetirməyi tələb edir (bax: Gün 7).

## 6.3. Məlumatın Həcmi Hədəfi

100M parametrli bir model üçün nə qədər məlumat lazımdır?

Ümumi qayda olaraq, LLM təlimində **"1 Parametrə 1-10 Token"** nisbəti tövsiyə olunur.

*   **Modelimiz:** 100 Milyon (100,000,000) Parametr.
*   **Hədəf Token Sayı (Minimum):** 100 Milyon Token.

Azərbaycan dilində bir token təxminən 5-6 simvola bərabərdir. 100 milyon token təxminən **500-600 milyon simvol** və ya **500-600 MB** xalis mətn deməkdir.

Bizim hədəfimiz **minimum 1 GB xalis mətn** toplamaq olacaq. Bu, modelin keyfiyyətini artırmaq üçün əlavə "qida" rolunu oynayacaq.

## 6.4. Günün Tapşırığı: Mənbə Siyahısının Hazırlanması

Bu günün tapşırığı, növbəti günlərdə Web Scraping edəcəyimiz **5-10 əsas veb-saytın URL-lərini** müəyyənləşdirməkdir.

1.  **Vikipediya:** Azərbaycan Vikipediyasının əsas səhifəsi.
2.  **Xəbər Saytı:** Bir neçə böyük xəbər portalının əsas səhifələri.
3.  **Rəsmi Sayt:** Məsələn, bir nazirliyin və ya universitetin saytı.

Bu URL-ləri bir faylda (məsələn, `urls.txt`) saxlayın. Sabah bu URL-lərdən məlumat çəkməyə başlayacağıq.
