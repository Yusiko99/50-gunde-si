# Gün 37: Modelin Qiymətləndirilməsi və Nəticələrin Təhlili 📈

## 37.1. LLM Qiymətləndirməsi

Təlimi bitirdik, modeli Ollama-ya yerləşdirdik. İndi isə modelimizin nə qədər yaxşı olduğunu obyektiv şəkildə qiymətləndirməliyik. LLM-lərin qiymətləndirilməsi iki əsas kateqoriyaya bölünür:

### A. Intrinsik Metrikalar (Daxili)

Bunlar modelin daxili xüsusiyyətlərini ölçür və təlim zamanı istifadə olunur:

1.  **Loss (İtki):** Modelin proqnozlaşdırma səhvi.
2.  **Perplexity (PPL):** Modelin nə qədər "çaşqın" olduğunu göstərən əsas metrika.

### B. Ekstrinsik Metrikalar (Xarici)

Bunlar modelin real dünya tapşırıqlarında nə qədər yaxşı işlədiyini ölçür:

1.  **BLEU/ROUGE:** Tərcümə və ya xülasə tapşırıqlarında istifadə olunur.
2.  **İnsan Qiymətləndirməsi:** Ən vacib metrika. İnsanlar modelin yaratdığı mətnin keyfiyyətini (axıcılıq, məntiqlilik, uyğunluq) qiymətləndirir.

## 37.2. Nəticələrin Təhlili

Sizin 134M parametrli modeliniz üçün gözlənilən nəticələr:

| Nəticə | Gözlənti | Səbəbi |
| :--- | :--- | :--- |
| **Axıcılıq** | Yüksək | Model Azərbaycan dilinin qrammatik quruluşunu (söz sırası, şəkilçilər) öyrənəcək. |
| **Məntiqlilik** | Orta | Kiçik model olduğu üçün uzun və mürəkkəb mətnlərdə məntiqi ardıcıllığı qorumaqda çətinlik çəkə bilər. |
| **Bilik** | Yalnız Korpus Bilikləri | Model yalnız sizin topladığınız korpusdakı məlumatları bilir. Korpusda olmayan mövzular haqqında cavab verə bilməyəcək. |
| **Halüsinasiya** | Orta | Bəzən model uydurma faktlar (halüsinasiya) yarada bilər. |

## 37.3. Modelin Təkmilləşdirilməsi Yolları

Əgər modelin nəticələri sizi qane etmirsə, aşağıdakı təkmilləşdirmə yollarını nəzərdən keçirə bilərsiniz:

1.  **Daha Çox Məlumat:** Korpusunuzun həcmini artırın. Məlumatın keyfiyyəti modelin keyfiyyətini birbaşa müəyyənləşdirir.
2.  **Daha Uzun Təlim:** Daha çox epoxa (dövr) təlim edin.
3.  **Hiperparametrlərin Tənzimlənməsi:** Öyrənmə sürətini (Learning Rate) və ya Dropout dərəcəsini dəyişdirin.
4.  **Daha Böyük Model:** Əgər resurslarınız imkan verərsə (məsələn, 12GB VRAM-lı T4), modelin ölçüsünü (n_embd, n_layer) artırın.

## 37.4. Günün Tapşırığı: Nümunə Test

Modelinizin Ollama-da yaratdığı ən azı 5 fərqli cavabı toplayın. Hər bir cavabı yuxarıdakı kriteriyalara əsasən qiymətləndirin və nəticələri qeyd edin. Bu, modelinizin güclü və zəif tərəflərini görməyə kömək edəcək.
