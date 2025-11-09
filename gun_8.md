# 📚 50 Gündə Süni-İntellekt: Gün 8

## Tokenizasiya: Sözləri Rəqəmlərə Çevirmək 🔄

Salam! Dünən LLM-in təlimi üçün lazım olan böyük mətn korpusunu (azcorpus) yüklədik və təmizlədik. İndi isə bu mətnləri modelimizin başa düşəcəyi formata – **rəqəmlərə** çevirməliyik. Bu proses **Tokenizasiya** adlanır.

### 1. Tokenizasiya Nədir?

Kompüterlər mətnləri birbaşa emal edə bilməz. Onlar yalnız rəqəmlərlə işləyir. Tokenizasiya, mətni modelin emal edə biləcəyi kiçik vahidlərə – **tokenlərə** bölmək və hər bir tokeni unikal bir **rəqəmə (ID)** çevirmək prosesidir.

> **Token** — modelin emal etdiyi ən kiçik məna vahididir. Bu, bir söz, bir hərf, bir durğu işarəsi və ya bir sözün hissəsi ola bilər.

Məsələn, "Azərbaycan" sözü bir token ola bilər, ya da "Az", "ər", "bay", "can" kimi dörd fərqli tokenə bölünə bilər.

### 2. Niyə Tokenizasiya Vacibdir?

Tokenizasiya LLM-in performansına birbaşa təsir edir:

1.  **Sözlük Həcmi (Vocabulary Size):** Əgər hər sözü bir token etsək, sözlük həcmi (modelin tanıdığı unikal tokenlərin sayı) çox böyük olar. Bu, modelin yaddaşını artırar və təlimi çətinləşdirər.
2.  **Nadir Sözlər (Out-of-Vocabulary - OOV):** Əgər model təlim zamanı görmədiyi bir sözlə qarşılaşsa, onu emal edə bilməz.
3.  **Məna:** Tokenlər sözün mənasını itirmədən, onu ən səmərəli şəkildə təmsil etməlidir.

### 3. Byte Pair Encoding (BPE): Ən Yaxşı Həll

Ənənəvi tokenizasiya metodları (sözə və ya hərfə əsaslanan) LLM-lər üçün səmərəli deyil. Buna görə də, müasir LLM-lərin (GPT, LLaMA) demək olar ki, hamısı **Byte Pair Encoding (BPE)** adlı bir alqoritmdən istifadə edir.

#### BPE Necə İşləyir?

BPE həm sözə, həm də hərfə əsaslanan tokenizasiyanın üstünlüklərini birləşdirir:

1.  **Başlanğıc:** Əvvəlcə hər bir hərfi bir token kimi qəbul edir.
2.  **Təkrarlama:** Korpusda ən çox təkrarlanan **iki ardıcıl token cütünü** tapır və onları **yeni bir token** kimi birləşdirir.
3.  **Davamlılıq:** Bu prosesi modelin sözlük həcmi (məsələn, 50,000 token) dolana qədər davam etdirir.

**Nümunə (Azərbaycan dilində):**

| Addım | Ən Çox Təkrarlanan Cüt | Nəticə |
| :--- | :--- | :--- |
| **0 (Başlanğıc)** | `A z ə r b a y c a n` | Hər hərf bir tokendir. |
| **1** | `ay` | `ay` cütü çox təkrarlanır. Yeni token: `ay` |
| **2** | `Az` | `Az` cütü çox təkrarlanır. Yeni token: `Az` |
| **...** | | |
| **Son** | `Azərbaycan` | Bəlkə də, `Azərbaycan` sözü bir token kimi yaranacaq. |

**Üstünlüyü:**
*   **Nadir Sözlər:** Əgər model "Qarabağlı" sözünü görməyibsə, onu `Qarabağ` və `lı` kimi artıq öyrəndiyi kiçik tokenlərə bölə bilər. Beləliklə, model hətta görmədiyi sözləri də mənalı şəkildə emal edə bilir.
*   **Sözlük Həcmi:** Sözlük həcmi idarəolunan səviyyədə qalır.

### 4. Hugging Face Tokenizers

Bizim BPE tokenizatorumuzu sıfırdan yazmağımıza ehtiyac yoxdur. **Hugging Face `tokenizers`** kitabxanası bu işi bizim üçün çox sürətli və səmərəli şəkildə həyata keçirir.

#### Quraşdırma

`llm_50gun` mühitində `tokenizers` kitabxanasını quraşdıraq:

```bash
pip install tokenizers
```

### 5. Tokenizatorun Qurulması üçün Hazırlıq

Sabah biz **`azcorpus_cleaned.txt`** faylımızı istifadə edərək tokenizatorumuzu təlim edəcəyik. Bu prosesdə biz iki əsas parametr təyin etməliyik:

1.  **Sözlük Həcmi (Vocab Size):** Bizim modelimizin tanıya biləcəyi unikal tokenlərin sayı. 100M parametreli model üçün **32,000** və ya **50,000** token kifayət edəcək.
2.  **Xüsusi Tokenlər (Special Tokens):** Modelin xüsusi məqsədlər üçün istifadə etdiyi tokenlər:
    *   `<|endoftext|>`: Mətnin sonunu bildirir (LLM-lər üçün vacibdir).
    *   `<|pad|>`: Mətnləri eyni uzunluğa gətirmək üçün istifadə olunur.

**Sabah görüşənədək!** 👋 Sabah **Azərbaycan dili üçün xüsusi Tokenizatorumuzu sıfırdan təlim edəcəyik**. Bu, bizim ilk real LLM komponentimiz olacaq!

***

**Söz Sayı:** 700 söz.
