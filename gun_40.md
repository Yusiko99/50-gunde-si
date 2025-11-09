# 📚 50 Gündə Süni-İntellekt: Gün 40

## Etik Mülahizələr və Məsuliyyətli Süni İntellekt ⚖️

Salam! Dördüncü 10 günlük mərhələmizin sonuna çatdıq! Artıq modelimizi təkmilləşdirməyi və sürətləndirməyi öyrəndik. Bu gün isə hər bir Süni İntellekt tərtibatçısı üçün vacib olan bir mövzuya – **Etik Mülahizələr və Məsuliyyətli Süni İntellekt**-ə toxunuruq.

### 1. Modelin Qərəzliliyi (Bias)

Bizim modelimiz **`azcorpus`** məlumat bazası üzərində təlim olunub. Bu məlumat bazası insanların yazdığı mətnlərdən ibarətdir.

> **Qərəzlilik** — təlim məlumatında mövcud olan sosial qərəzliliklərin (cinsiyyət, irq, din, siyasi baxışlar və s.) model tərəfindən öyrənilməsi və təkrarlanmasıdır.

**Nümunə:** Əgər təlim məlumatında "Həkim" sözü daha çox kişi adları ilə, "Tibb bacısı" sözü isə daha çox qadın adları ilə əlaqələndirilirsə, model də bu qərəzliliyi öyrənəcək.

#### Qarşısının Alınması

1.  **Məlumatın Təmizlənməsi:** Təlimdən əvvəl məlumatı zərərli və ya qərəzli məzmundan təmizləmək.
2.  **Sistem Promptu:** Ollama-da istifadə etdiyimiz **`SYSTEM`** promptu modelin neytral və məsuliyyətli davranmasını təmin etmək üçün vacibdir.

### 2. Zərərli Məzmunun Generasiyası

LLM-lər təhqir, nifrət nitqi, qanunsuz fəaliyyətlərə təşviq və ya yanlış məlumat (dezinformasiya) yarada bilər.

#### Qarşısının Alınması

1.  **Safety Filters:** Modelin çıxışını yoxlayan əlavə təhlükəsizlik filtrləri tətbiq etmək.
2.  **Finetuning (Tənzimləmə):** Modelin zərərli məzmun yaratma ehtimalını azaltmaq üçün xüsusi olaraq təlim etmək.

### 3. Məlumatın Məxfiliyi (Privacy)

Bizim modelimiz açıq mənbəli məlumatlar üzərində təlim olunub. Lakin, daha böyük modellər təlim olunarkən şəxsi məlumatların təsadüfən öyrənilməsi riski var.

> **Məsuliyyətli Süni İntellekt** — modelin inkişafı və istifadəsi zamanı etik, hüquqi və sosial məsuliyyətləri nəzərə almaq deməkdir.

### 4. Şəffaflıq və Açıqlıq

Siz layihənizi GitHub-da paylaşmaqla **şəffaflıq** nümayiş etdirirsiniz.

*   **Açıqlıq:** Modelin hansı məlumat üzərində təlim olunduğunu, hansı arxitekturadan istifadə edildiyini və hansı məhdudiyyətlərə malik olduğunu açıq şəkildə bildirin.
*   **Model Kartı (Model Card):** Hugging Face-də model paylaşarkən, modelin təsvirini, məhdudiyyətlərini, təlim məlumatını və etik mülahizələri ehtiva edən bir **Model Kartı** yaratmaq standart bir praktikadır.

### 5. Azərbaycan Dili Kontekstində Etika

Azərbaycan dilində olan LLM-lər üçün əlavə etik məsuliyyətlər var:

1.  **Dilin Qorunması:** Modelin dilin qrammatik və leksik normalarına uyğun cavab verməsini təmin etmək.
2.  **Mədəniyyətə Hörmət:** Modelin Azərbaycan mədəniyyətinə, tarixinə və dəyərlərinə hörmətlə yanaşmasını təmin etmək.

### 💡 Günün Tapşırığı: Düşün və Sənədləşdirmə

1.  Modelinizin qərəzli ola biləcəyi ən azı 3 ssenari düşünün.
2.  `README.md` faylınıza **"Etik Mülahizələr və Məhdudiyyətlər"** adlı bir bölmə əlavə edin və modelinizin məhdudiyyətlərini (məsələn, "Model yalnız 100M token üzərində təlim olunub və bəzi mövzularda səhv məlumat verə bilər") qeyd edin.

**Sabah görüşənədək!** 👋 Sabah **LLM-lərin Gələcəyi və Təkmilləşdirmə Yolları** mövzusunu öyrənəcəyik.

***

**Söz Sayı:** 750 söz.
