# Gün 13: Transformer: LLM-lərin Beyni 🧠

## 13.1. Transformer Arxitekturası Nədir?

2017-ci ildə Google tərəfindən təqdim edilən **Transformer** arxitekturası, Böyük Dil Modellərinin (LLM) əsasını təşkil edir. Bu arxitektura, əvvəlki modellərdən (RNN, LSTM) fərqli olaraq, mətnin emalında **ardıcıllıqdan asılılığı** aradan qaldırdı və **paralel hesablama** imkanlarını kəskin şəkildə artırdı.

Transformer-in əsas üstünlüyü, mətnin istənilən hissəsinə eyni anda baxa bilməsidir.

## 13.2. Transformer-in Əsas Komponentləri

Transformer iki əsas hissədən ibarətdir:

| Komponent | Məqsəd | Bizim Modelimizdəki Rolu |
| :--- | :--- | :--- |
| **Encoder (Kodlayıcı)** | Giriş mətnini (cümələri) oxuyur və onun mənasını başa düşür. | Bizim modelimizdə istifadə **edilmir**. |
| **Decoder (Dekodlayıcı)** | Encoder-dən gələn məlumatı istifadə edərək çıxış mətnini (cavabı) yaradır. | **Bizim modelimizin əsasını təşkil edir.** |

Bizim **GPT (Generative Pre-trained Transformer)** modelimiz, adından da göründüyü kimi, yalnız **Decoder** hissəsindən istifadə edir. Bu, modelin **bir sonrakı tokeni proqnozlaşdırmaq** üçün nəzərdə tutulmuş bir arxitekturadır.

## 13.3. Decoder-in Daxili Quruluşu

Decoder-in əsasında iki vacib mexanizm dayanır:

1.  **Masked Multi-Head Attention (Maskalanmış Çoxbaşlı Diqqət):** Mətnin bir hissəsinə baxarkən, modelin hansı sözlərə daha çox diqqət yetirməli olduğunu müəyyənləşdirir. **"Maskalanmış"** olması o deməkdir ki, model proqnozlaşdırdığı sözdən sonrakı sözlərə baxa bilməz. Bu, modelin "fırıldaqçılıq" etməsinin qarşısını alır.
2.  **Feed-Forward Network (İrəli-Ötürmə Şəbəkəsi):** Diqqət mexanizmindən gələn məlumatı emal edir və modelin öyrənmə qabiliyyətini artırır.

Bu iki blok ardıcıl olaraq bir neçə dəfə (bizim 100M modelimizdə 12 dəfə) təkrarlanır.

## 13.4. NanoGPT-yə Giriş

Biz modelimizi Andrej Karpathy tərəfindən yaradılmış **NanoGPT** layihəsinin sadələşdirilmiş versiyasına əsaslanaraq quracağıq. NanoGPT, GPT-nin əsas prinsiplərini **minimum kodla** izah etmək üçün nəzərdə tutulmuşdur.

**NanoGPT-nin Əsas Xüsusiyyətləri:**

*   **Sadəlik:** Mürəkkəb optimallaşdırmalar olmadan, təmiz PyTorch kodu.
*   **Öyrənməyə Fokus:** Hər bir hissənin funksiyasını asanlıqla anlamağa imkan verir.

Biz NanoGPT-nin arxitekturasını götürəcək, onu Azərbaycan dilinə uyğunlaşdıracaq və RTX 2050 üçün optimallaşdıracağıq.

**Modelimizin Parametrləri (100M hədəfi üçün):**

| Parametr | Dəyər | İzahı |
| :--- | :--- | :--- |
| **Block Size (Context Length)** | 256 | Modelin bir dəfəyə emal edə biləcəyi maksimum token sayı. |
| **Embedding Dimension (n_embd)** | 768 | Hər bir tokenin rəqəmsal təsvirinin ölçüsü. |
| **Number of Heads (n_head)** | 12 | Multi-Head Attention-dakı "baş" sayı. |
| **Number of Layers (n_layer)** | 12 | Təkrarlanan Transformer Bloklarının sayı. |
| **Vocabulary Size (vocab_size)** | 32000 | Tokenizatorumuzun lüğət ölçüsü. |

Bu konfiqurasiya ilə modelimizin parametr sayı təxminən **100 milyon** olacaq.

**Gündəlik Tapşırıq:** Transformer arxitekturası haqqında qısa bir video izləyin. Xüsusilə **"Masked Self-Attention"** anlayışını dərindən başa düşməyə çalışın.
