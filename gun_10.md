# Gün 10: Tokenizasiya I: Sözləri Rəqəmlərə Çevirmək 🔢

## 10.1. Tokenizasiyanın Zəruriliyi

Neyron şəbəkələri mətnlə deyil, yalnız **rəqəmlərlə** işləyə bilər. **Tokenizasiya** prosesi, insan dilindəki mətnin modelin başa düşəcəyi rəqəmsal ardıcıllıqlara çevrilməsidir.

**Məntiq:** Hər bir unikal söz, sözün hissəsi və ya simvol (token) lüğətdə (Vocabulary) bir unikal tam ədədə (ID) uyğun gəlir. Model bu ID-ləri giriş kimi qəbul edir və çıxışda növbəti tokenin ID-sini proqnozlaşdırır.

## 10.2. Tokenizator Növləri

LLM-lərdə ən çox istifadə olunan tokenizator növləri aşağıdakılardır:

| Növ | Məntiqi Əsas | Üstünlüyü |
| :--- | :--- | :--- |
| **Word-based** | Hər söz bir tokendir. | Sadədir. |
| **Character-based** | Hər simvol bir tokendir. | Lüğət kiçikdir, lakin ardıcıllıqlar çox uzundur. |
| **Subword-based (BPE)** | Sözləri tez-tez təkrarlanan alt-vahidlərinə (subwords) bölür. | **LLM-lər üçün standartdır.** Lüğət ölçüsü ilə ardıcıllıq uzunluğu arasında optimal balans yaradır. |

Bizim modelimiz üçün **Subword-based (BPE - Byte Pair Encoding)** tokenizatoru istifadə ediləcək.

## 10.3. Byte Pair Encoding (BPE) Məntiqi

BPE alqoritmi aşağıdakı məntiqə əsaslanır:

1.  **Başlanğıc:** Bütün mətn simvollara bölünür.
2.  **Təkrarlama:** Korpusda ən çox təkrarlanan bitişik simvol cütü (və ya token cütü) tapılır.
3.  **Birləşdirmə:** Tapılan cüt yeni bir token kimi lüğətə əlavə edilir və mətndəki bütün rast gəlinən yerlərdə bu yeni tokenlə əvəz edilir.
4.  **Son:** Bu proses, lüğət ölçüsü (Vocabulary Size) əvvəlcədən təyin edilmiş həddə çatana qədər təkrarlanır.

**Nümunə:** "Azərbaycan" sözü.

| Addım | Ən Çox Təkrarlanan Cüt | Nəticə |
| :--- | :--- | :--- |
| **Başlanğıc** | `A z ə r b a y c a n` | Simvollar |
| **1** | `az` | `az` tokeni yaranır. |
| **2** | `an` | `an` tokeni yaranır. |
| **...** | | Yekunda: `Azər` + `bay` + `can` kimi alt-sözlərə bölünə bilər. |

**Məntiq:** BPE, tez-tez rast gəlinən sözləri (məsələn, "kitab") tək bir token kimi, nadir sözləri (məsələn, "kitabxanaçılıq") isə bir neçə tokenin birləşməsi kimi kodlaşdırır. Bu, lüğətin ölçüsünü idarə etməyə və naməlum sözlərin (OOV - Out-of-Vocabulary) qarşısını almağa kömək edir.

## 10.4. Lüğət Ölçüsünün Seçilməsi

LLM-lər üçün lüğət ölçüsü adətən 30,000 ilə 50,000 arasında seçilir. Bizim 100M parametrli modelimiz üçün **32,000 tokenlik** bir lüğət ölçüsü seçiləcək.

**Məntiq:** Lüğət nə qədər böyük olsa, model bir sözü bir tokenlə ifadə etməyə o qədər yaxın olar, lakin bu, modelin yaddaş tələbini artırar. 32,000 optimal bir balans təmin edir.

## 10.5. Günün Tapşırığı: Tokenizator Kitabxanasının Quraşdırılması

Növbəti gün BPE tokenizatorunu təlim etmək üçün Hugging Face-in **`tokenizers`** kitabxanasından istifadə ediləcək. Bu kitabxana Rust dilində yazıldığı üçün çox sürətlidir.

```bash
pip install tokenizers
```
