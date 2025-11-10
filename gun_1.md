# Gün 1: Giriş: Süni İntellektə İlk Addım 🚀

## 1.1. Süni İntellekt (Sİ) və Böyük Dil Modelləri (LLM)

Bu təlim modulu, Süni İntellekt (Sİ) sahəsində fundamental bir layihənin – **Böyük Dil Modelinin (LLM)** – sıfırdan inşasına həsr olunmuşdur. Modulun hədəfi, Azərbaycan dilində ünsiyyət qura bilən, **100 Milyon (100M)** parametrli, yüngül və effektiv bir LLM arxitekturasının necə qurulduğunu dərindən öyrətməkdir.

**Süni İntellekt (Sİ)**, maşınların insan zəkasına xas olan öyrənmə, qərar qəbul etmə və problem həll etmə kimi vəzifələri yerinə yetirmə qabiliyyətidir.

**Böyük Dil Modelləri (LLM)** isə Sİ-nin bir alt sahəsi olub, milyardlarla sözdən ibarət mətn məlumatları üzərində təlim keçmiş, insan dilini anlamaq və mətn yaratmaq üçün nəzərdə tutulmuş neyron şəbəkələridir.

## 1.2. Sıfırdan İnşa Metodologiyası

Bu təlimdə **hazır modellərin tənzimlənməsi (Fine-tuning)** və ya **məlumat bazası ilə zənginləşdirilmiş generasiya (RAG)** kimi metodlardan istifadə edilməyəcək. Əsas fokus, modelin bütün komponentlərinin **fundamental səviyyədə** başa düşülməsinə yönəldilmişdir. Bu metodologiya aşağıdakı əsas mərhələləri əhatə edir:

1.  **Sıfırdan Korpus İnşası:** Təlim üçün lazım olan Azərbaycan dili mətn korpusunun mənbələrdən toplanması və təmizlənməsi.
2.  **Model Arxitekturasının Qurulması:** Transformer arxitekturasının hər bir blokunun (Attention, Feed-Forward) PyTorch-da kodlaşdırılması.
3.  **Sıfırdan Təlim:** Modelin toplanmış korpus üzərində ilkin təlimi.

Bu yanaşma, LLM-lərin **daxili iş prinsipini** və **riyazi əsaslarını** mənimsəməyə imkan verir.

## 1.3. Texniki Məhdudiyyətlər və Optimallaşdırma

Modelin təlimi üçün məhdud VRAM (Video RAM) resursu (məsələn, **4GB VRAM**) nəzərdə tutulur. Bu texniki məhdudiyyət, 100M parametrli modelin uğurlu təlimi üçün **kritik optimallaşdırmaların** tətbiqini zəruri edir.

| Texnika | Məqsəd | Məntiqi Əsas |
| :--- | :--- | :--- |
| **Mixed Precision (FP16)** | VRAM istifadəsini **50% azaltmaq**. | Modelin çəkilərini və qradiyentlərini 32-bit (FP32) əvəzinə 16-bit (FP16) dəqiqlikdə saxlamaqla, hər bir parametr üçün tələb olunan yaddaş həcmi yarıya enir. |
| **Gradient Accumulation** | **Effektiv Batch Size-ı artırmaq**. | Qradiyentləri bir neçə kiçik "mini-batch" üzərində toplayıb, yalnız sonda modelin çəkilərini yeniləmək. Bu, VRAM-ı doldurmadan daha böyük bir Batch Size-ın təsirini simulyasiya edir. |

Bu optimallaşdırmalar, məhdud resurslar şəraitində belə, böyük modellərin təlimini mümkün edən əsas vasitələrdir.

## 1.4. Əsas Terminologiya

LLM təlimi prosesinə başlamazdan əvvəl, əsas terminologiyanın mənimsənilməsi vacibdir.

| Termin | Azərbaycan Dilində | Məntiqi İzahı |
| :--- | :--- | :--- |
| **VRAM** | Video RAM | GPU-nun yaddaşı. Təlim zamanı modelin çəkiləri, qradiyentləri və aralıq hesablamalar burada saxlanılır. Məhdud VRAM (4GB) optimallaşdırma tələb edir. |
| **Parameter** | Parametr | Modelin öyrəndiyi dəyişənlərin sayı. Modelin biliyini və mürəkkəbliyini müəyyən edir. |
| **Token** | Token | Mətnin model tərəfindən emal edilən ən kiçik vahidi (söz, sözün hissəsi və ya simvol). |
| **Corpus** | Korpus | Təlim üçün istifadə olunan, dilin bütün xüsusiyyətlərini əks etdirən böyük mətn toplusu. |
| **Batch Size** | Paket Ölçüsü | Bir dəfəyə GPU-ya göndərilən məlumat nümunələrinin sayı. Təlimin sürətinə və VRAM tələbinə birbaşa təsir edir. |
| **Loss** | İtki | Modelin proqnozlaşdırma səhvinin ölçüsü. Təlimin məqsədi bu dəyəri minimuma endirməkdir. |

**Qeyd:** Təlimin ilk mərhələsi (Gün 6-dan başlayaraq) **Azərbaycan dili korpusunun sıfırdan inşasına** həsr olunacaq.
