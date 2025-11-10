# Gün 1: Giriş: Süni İntellektə İlk Addım 🚀

## 1.1. Süni İntellekt (Sİ) və Böyük Dil Modelləri (LLM) Nədir?

Bu gün, Süni İntellekt (Sİ) dünyasına atdığımız böyük səyahətin ilk günüdür. 50 gün ərzində biz birlikdə sıfırdan başlayaraq, Azərbaycan dilində danışa bilən və sizinlə ünsiyyət qura bilən bir chatbotun – yəni **Böyük Dil Modelinin (LLM)** – necə yaradıldığını öyrənəcəyik.

**Süni İntellekt (Sİ)**, maşınların insan zəkasına xas olan vəzifələri (öyrənmə, qərar qəbul etmə, problem həll etmə) yerinə yetirmə qabiliyyətidir.

**Böyük Dil Modelləri (LLM)** isə Sİ-nin bir növüdür. Onlar **milyardlarla** sözdən ibarət mətn məlumatları üzərində təlim keçmiş, insan dilini anlamaq və yaratmaq üçün nəzərdə tutulmuş nəhəng neyron şəbəkələridir. Bizim məqsədimiz **100 milyon (100M)** parametrli, yüngül, lakin güclü bir LLM yaratmaqdır.

## 1.2. Niyə Sıfırdan Başlayırıq?

Siz soruşa bilərsiniz: "Hazır modellər varkən niyə sıfırdan başlayırıq?"

Bu kitabın əsas məqsədi **LLM-lərin necə işlədiyini dərindən anlamaqdır**. Biz **Fine-tuning (Tənzimləmə)** və ya **RAG (Retrieval-Augmented Generation)** kimi hazır metodlardan istifadə etməyəcəyik. Əksinə, biz:

1.  **Sıfırdan Dataset İnşası:** Azərbaycan dilində məlumatları **özümüz** toplayıb təmizləyəcəyik.
2.  **Sıfırdan Model Arxitekturası:** Modelin hər bir hissəsini (Transformer, Attention) **özümüz** kodlayacağıq.
3.  **Sıfırdan Təlim:** Modelimizi sıfırdan təlim edəcəyik.

Bu yanaşma sizə LLM-lərin **əsl iş prinsipini** öyrədəcək.

## 1.3. Sizin Cihazınız: RTX 2050 (4GB VRAM) ilə Təlim

Sizdə **NVIDIA RTX 2050 (4GB VRAM)** kartının olduğunu nəzərə alaraq, bu modelin təlimi üçün bəzi **kritik optimallaşdırmalar** tətbiq etməliyik.

**Yaxşı Xəbər:** Bəli, 100M parametrli modeli bu kartla təlim etmək **tamamilə mümkündür**.

**Pis Xəbər:** Bu, VRAM (Video RAM) baxımından çox məhdud bir resursdur. Hər hansı bir səhv, proqramın "Yaddaşdan Kənar" (Out-of-Memory - OOM) xətası ilə dayanmasına səbəb ola bilər.

Buna görə də, kitab boyunca iki əsas optimallaşdırma texnikasına diqqət yetirəcəyik:

| Texnika | Məqsəd | Necə İşləyir |
| :--- | :--- | :--- |
| **Mixed Precision (FP16)** | VRAM istifadəsini **50% azaltmaq**. | Ədədlərin dəqiqliyini 32 bitdən (FP32) 16 bitə (FP16) endirir. Bu, modelin çəkilərini və qradiyentlərini yaddaşda daha az yer tutmağa məcbur edir. |
| **Gradient Accumulation (Qradiyent Yığımı)** | **Effektiv Batch Size-ı artırmaq**. | Modelin çəkilərini yeniləmədən əvvəl bir neçə kiçik "mini-batch" üzərində qradiyentləri toplayır. Bu, VRAM-ı doldurmadan daha böyük bir Batch Size-ın təsirini simulyasiya etməyə imkan verir. |

Bu texnikalar sayəsində, 4GB VRAM-a baxmayaraq, 100M parametrli modelin təlimini uğurla başa çatdıra biləcəyik.

## 1.4. Günün Tapşırığı: Terminologiya ilə Tanışlıq

Bu günün əsas tapşırığı, LLM dünyasında istifadə olunan əsas terminologiya ilə tanış olmaqdır. Bu terminləri başa düşmədən irəliləmək çətin olacaq.

| Termin | Azərbaycan Dilində | İzahı |
| :--- | :--- | :--- |
| **LLM** | Böyük Dil Modeli | Mətn yaratmaq və anlamaq üçün nəhəng neyron şəbəkəsi. |
| **VRAM** | Video RAM | GPU-nun yaddaşı. Təlim zamanı modelin çəkiləri və qradiyentləri burada saxlanılır. **Sizin üçün 4GB.** |
| **Parameter** | Parametr | Modelin öyrəndiyi dəyişənlərin sayı. **Bizim modelimiz 100M olacaq.** |
| **Token** | Token | Mətnin ən kiçik vahidi (söz, sözün hissəsi və ya simvol). |
| **Corpus** | Korpus | Təlim üçün istifadə olunan böyük mətn toplusu. |
| **Batch Size** | Paket Ölçüsü | Bir dəfəyə GPU-ya göndərilən məlumat nümunələrinin sayı. |
| **Epoch** | Epoxa | Bütün korpusun model tərəfindən bir dəfə oxunması. |
| **Loss** | İtki | Modelin nə qədər səhv etdiyini göstərən rəqəm. Məqsəd bu rəqəmi azaltmaqdır. |

**Unutmayın:** Bizim ilk böyük addımımız **Gün 6-da** başlayacaq **sıfırdan Azərbaycan dili korpusu yaratmaq** olacaq. Buna görə də, növbəti günlərdə Python və iş mühitini hazırlayarkən, bu məqsədi unutmayın.

**Gündəlik Tapşırıq:** Yuxarıdakı terminləri öz sözlərinizlə izah etməyə çalışın. Bu, öyrənmə prosesini sürətləndirəcək.
