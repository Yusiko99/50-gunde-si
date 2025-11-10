# Gün 6: Dataset İnşası I: Məlumat Mənbələrinin Təyini 🗺️

## 6.1. Korpusun Funksional Əhəmiyyəti

**Korpus (Corpus)** – təlim üçün istifadə olunan böyük həcmli mətn toplusudur. LLM-in təlimində korpus, modelin **dilin qrammatik, sintaktik və semantik qaydalarını** öyrəndiyi əsas "qida" rolunu oynayır. Korpusun keyfiyyəti və müxtəlifliyi, modelin yekun performansını birbaşa müəyyənləşdirir.

Bu təlimdə, modelin dil biliyini **tamamilə sıfırdan** qurmaq üçün, hazır korpuslardan istifadə edilməyəcək. Əksinə, korpusun inşası prosesi mərhələli şəkildə öyrədiləcək.

## 6.2. Azərbaycan Dili Mənbələrinin Seçilməsi

Azərbaycan dili kimi **aşağı resurslu (low-resource)** dillər üçün keyfiyyətli və böyük həcmli mətn mənbələri tapmaq, ingilis dili ilə müqayisədə daha çətindir. Buna görə də, mənbələrin seçimi modelin **müxtəlif mövzularda** və **müxtəlif üslublarda** öyrənməsini təmin etməlidir.

| Mənbə Növü | Məntiqi Əhəmiyyəti | Təmsil Etdiyi Üslub |
| :--- | :--- | :--- |
| **Vikipediya** | **Elmi və faktiki biliklərin** əsas mənbəyi. Modelin neytral və ensiklopedik tonu öyrənməsini təmin edir. | Rəsmi, Neytral |
| **Xəbər Saytları** | **Aktual hadisələr və terminologiya.** Siyasi, iqtisadi və idman leksikonunu təmin edir. | Jurnalistik, Aktual |
| **Rəsmi Sənədlər** | **Hüquqi və normativ dilin** strukturunu öyrədir. | Hüquqi, Formal |
| **Ədəbiyyat** | **Bədii və emosional dilin** zənginliyini və mürəkkəb cümlə quruluşlarını öyrədir. | Bədii, Emosional |
| **Forumlar/Bloqlar** | **Danışıq dilini, jarqonları** və qeyri-rəsmi üslubu təmin edir. | Qeyri-rəsmi, Danışıq |

## 6.3. Məlumatın Həcmi və Parametr Nisbəti

Modelin təlimi üçün tələb olunan məlumatın həcmi, modelin parametr sayına əsasən müəyyən edilir. LLM təlimində ümumi qəbul edilmiş nisbət **"1 Parametrə 1-10 Token"** nisbətidir.

*   **Model Parametri:** 100 Milyon.
*   **Minimum Hədəf Token Sayı:** 100 Milyon Token.

Bu, təxminən **500-600 MB xalis mətn** deməkdir. Lakin modelin keyfiyyətini artırmaq və təlimi sabitləşdirmək üçün **minimum 1 GB xalis mətn** toplanması tövsiyə olunur.

**Məntiq:** Modelin hər bir parametrinin effektiv şəkildə öyrənməsi üçün, hər bir parametrə kifayət qədər məlumat (token) təqdim edilməlidir.

## 6.4. Günün Tapşırığı: Mənbə Siyahısının Hazırlanması

Növbəti mərhələdə istifadə olunacaq **Web Scraping** prosesi üçün ən azı 5-10 müxtəlif və etibarlı Azərbaycan dili veb-saytının URL-ləri müəyyənləşdirilməlidir. Bu URL-lər bir faylda (`urls.txt`) saxlanılmalıdır.
