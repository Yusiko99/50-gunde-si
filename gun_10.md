# Gün 10: Tokenizasiya I: Sözləri Rəqəmlərə Çevirmək 🔢

## 10.1. Niyə Tokenizasiya?

İndiyə qədər biz təmiz və normallaşdırılmış mətn korpusu yaratdıq. Lakin kompüterlər və neyron şəbəkələr mətnlə deyil, **rəqəmlərlə** işləyir. **Tokenizasiya** prosesi mətnimizi modelin başa düşəcəyi rəqəmlər ardıcıllığına çevirməkdir.

**Token** – mətnin ən kiçik mənalı vahididir. Bu, bir söz, bir simvol və ya bir sözün hissəsi ola bilər.

**Vocabulary (Lüğət)** – korpusumuzda rast gəlinən bütün unikal tokenlərin siyahısıdır. Hər bir tokenin bu lüğətdə özünəməxsus bir **ID (İdentifikator)** nömrəsi var.

## 10.2. Byte Pair Encoding (BPE) Nədir?

LLM-lərdə ən çox istifadə olunan tokenizasiya üsulu **Byte Pair Encoding (BPE)**-dir.

**BPE-nin Əsas Prinsipi:**

1.  **Başlanğıc:** Bütün mətn simvollara bölünür (məsələn, "Azərbaycan" -> \['A', 'z', 'ə', 'r', 'b', 'a', 'y', 'c', 'a', 'n']).
2.  **Təkrarlama:** Ən çox təkrarlanan ardıcıl simvol cütləri tapılır və yeni bir token kimi lüğətə əlavə olunur.
3.  **Birləşdirmə:** Bu yeni tokenlər mətndəki cütləri əvəz edir.
4.  **Son:** Bu proses, ya lüğətin ölçüsü (məsələn, 32000 token) müəyyən bir həddə çatana qədər, ya da ən çox təkrarlanan cütlüyün sayı çox az olana qədər davam edir.

**Niyə BPE?**

*   **Sözləri Qoruyur:** Tez-tez rast gəlinən sözlər bir token kimi qalır (məsələn, "Azərbaycan").
*   **Nadir Sözləri Həll Edir:** Nadir və ya yeni sözlər (məsələn, "kvantlaşdırma") hissələrə bölünür (məsələn, \['kvant', 'laş', 'dır', 'ma']). Bu, modelin hər bir sözü görməsə belə, onun hissələrini tanımağa imkan verir.

## 10.3. Azərbaycan Dili üçün Tokenizasiya

Azərbaycan dili **aqqlütinativ** (iltisaqi) bir dildir. Yəni, sözlərə çoxlu sayda şəkilçilər qoşulur (məsələn, "kitablarımızdakılardan").

BPE bu cür dillər üçün çox uyğundur, çünki:

*   **Kök Sözlər:** "kitab" kimi kök sözlər tək token kimi qalır.
*   **Şəkilçilər:** "-larımız", "-dakı", "-lardan" kimi şəkilçilər ayrıca tokenlər kimi öyrənilir.

Bu, modelin kiçik bir lüğətlə belə sonsuz sayda söz kombinasiyasını anlamaq qabiliyyətini artırır.

## 10.4. Günün Tapşırığı: Tokenizatorun Təliminə Hazırlıq

Sabah biz Python-un `tokenizers` kitabxanasından istifadə edərək BPE tokenizatorumuzu təlim edəcəyik. Bu günün tapşırığı isə bu proses üçün lazım olan kitabxanaları quraşdırmaqdır.

**Terminalda icra edin:**

```bash
# Tokenizatorun təlimi üçün əsas kitabxana
pip install tokenizers
```

**Qeyd:** Bizim LLM modelimiz üçün ən optimal lüğət ölçüsü (Vocabulary Size) təxminən **32000** olacaq. Bu rəqəm, dilin zənginliyini qorumaq və modelin yaddaş tələbini minimuma endirmək üçün yaxşı bir tarazlıqdır.
