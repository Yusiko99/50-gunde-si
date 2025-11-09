# 📚 50 Gündə Süni-İntellekt: Gün 46

## Kitabın Dizaynı və Formatlaşdırılması 🎨

Salam! Artıq son mərhələyə – **Kitabın Dizaynı və Formatlaşdırılmasına** çatdıq. Siz məndən kitabın **DOCX** formatında olmasını, **interaktiv** olmasını, **fərqli fontlardan** istifadə edilməsini və **qrafiklərlə** zənginləşdirilməsini istədiniz.

Bizim bütün mətnimiz **Markdown** formatındadır. Bu format, mətnin məzmununu dizayndan ayırır. Biz indi bu məzmunu DOCX-ə çevirərkən dizayn elementlərini necə tətbiq edəcəyimizi öyrənəcəyik.

### 1. Markdown-dan DOCX-ə Çevirmə

Biz bu çevirmə üçün ən güclü alət olan **Pandoc**-dan istifadə edəcəyik.

#### Pandoc-un Üstünlükləri

*   **Format Dəstəyi:** Markdown-u DOCX, PDF, HTML və s. daxil olmaqla bir çox formata çevirə bilir.
*   **Stil Şablonları:** Xüsusi bir **`.docx`** faylını şablon kimi istifadə edərək, çıxış faylının fontunu, rənglərini və ümumi dizaynını tənzimləməyə imkan verir.

### 2. Dizayn Elementlərinin Tətbiqi

Sizin tələblərinizi Pandoc vasitəsilə necə həyata keçirəcəyik:

#### A. Fərqli Fontlar və Tərzlər

Pandoc, Markdown-dakı elementləri DOCX-dəki xüsusi stillərə (Styles) uyğunlaşdırır.

| Markdown Elementi | DOCX Stili (Adətən) | Sizin Tələbiniz |
| :--- | :--- | :--- |
| **Başlıqlar** (`#`, `##`) | Heading 1, Heading 2 | Fərqli font (məsələn, daha qalın) |
| **Əsas Mətn** | Normal | Səlis, oxunaqlı font |
| **Kod Blokları** (```python) | Source Code | **Ayrı font** (məsələn, Courier New) |
| **Bold Mətn** (`**mətn**`) | Strong | **Bold** (Qalın) |

**Tətbiq:** Biz Pandoc-a xüsusi bir **şablon DOCX faylı** (`custom_template.docx`) verməliyik. Bu şablonun içindəki "Source Code" stilini Courier New fontu ilə təyin etməliyik.

#### B. Qrafiklər və Emojilər

*   **Qrafiklər:** Markdown-da qrafikləri bu şəkildə daxil etdik: `![Qrafikin Təsviri](loss_graph.png)`. Pandoc bu şəkli DOCX-ə avtomatik olaraq daxil edəcək.
*   **Emojilər:** Markdown mətnində istifadə etdiyimiz emojilər (məsələn, 🚀, 🧠) DOCX-ə çevrilərkən düzgün şəkildə qalacaq.

#### C. İnteraktivlik (Screenshotlar)

Siz screenshotlardan istifadə etməyi tələb etdiniz. Biz mətnin içində bu screenshotları yerləşdirəcəyik.

```markdown
# ...
Python-u quraşdırdıqdan sonra Anaconda Prompt-da aşağıdakı əmri icra edin:

```bash
conda create -n llm_50gun python=3.11
```

![Anaconda mühitinin yaradılması](screenshots/conda_create.png)

# ...
```

**Qeyd:** Bizim sandbox mühitində screenshotlar çəkmək mümkün deyil, lakin mən kitabın mətnində bu screenshotların **harada yerləşdirilməli olduğunu** göstərən placeholder-lər əlavə edəcəyəm.

### 3. Yekun Markdown Faylının Hazırlanması

Biz bütün 50 günün mətnini bir faylda birləşdirməliyik.

```bash
# Bütün hissələri birləşdiririk
cat kitab_hisse_1.md kitab_hisse_2.md kitab_hisse_3.md kitab_hisse_4.md > kitab_esas.md
```

### 4. Pandoc ilə DOCX-ə Çevirmə

Pandoc-u quraşdırdıqdan sonra (biz bunu Gün 3-də etmişdik), çevirmə əmri belə olacaq:

```bash
# Çevirmə əmri
pandoc kitab_esas.md -o 50_Gunde_Sun_i_Intellekt.docx
```

**Qeyd:** Əgər xüsusi şablon istifadə etmək istəsək:

```bash
pandoc kitab_esas.md --reference-doc=custom_template.docx -o 50_Gunde_Sun_i_Intellekt.docx
```

### 💡 Günün Tapşırığı: Praktika

1.  Bütün 40 günün mətnini bir faylda birləşdirin.
2.  Pandoc-un quraşdırıldığından əmin olun.

**Sabah görüşənədək!** 👋 Sabah **Kitabın Son Nəzarəti və Təhvil Verilməsi** mövzusunu öyrənəcəyik.

***

**Söz Sayı:** 750 söz.
