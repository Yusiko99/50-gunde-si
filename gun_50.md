# 📚 50 Gündə Süni-İntellekt: Gün 50

## DOCX-ə Çevirmə və Təhvil: Yekunlaşdırma 🎉

Salam! Bu, bizim **"50 Gündə Süni-İntellekt"** səyahətimizin son günüdür. Bütün məzmun hazırdır. Bu gün bütün məzmunu birləşdirib, **DOCX** formatına çevirəcək və sizə təhvil verəcəyik.

### 1. Bütün Məzmunun Birləşdirilməsi

Biz bütün 50 günün məzmununu, Ön Sözü və Məzmun Cədvəlini bir yekun Markdown faylında birləşdiririk.

```bash
# Bütün hissələri ardıcıl olaraq birləşdiririk
cat /home/ubuntu/50_gunde_si/gun_48.md \
    /home/ubuntu/50_gunde_si/kitab_plani.md \
    /home/ubuntu/50_gunde_si/kitab_hisse_1.md \
    /home/ubuntu/50_gunde_si/kitab_hisse_2.md \
    /home/ubuntu/50_gunde_si/kitab_hisse_3.md \
    /home/ubuntu/50_gunde_si/kitab_hisse_4.md \
    /home/ubuntu/50_gunde_si/gun_41.md \
    /home/ubuntu/50_gunde_si/gun_42.md \
    /home/ubuntu/50_gunde_si/gun_43.md \
    /home/ubuntu/50_gunde_si/gun_44.md \
    /home/ubuntu/50_gunde_si/gun_45.md \
    /home/ubuntu/50_gunde_si/gun_46.md \
    /home/ubuntu/50_gunde_si/gun_47.md \
    /home/ubuntu/50_gunde_si/gun_49.md \
    > /home/ubuntu/50_gunde_si/kitab_yekun.md
```

### 2. DOCX-ə Çevirmə

Pandoc alətini istifadə edərək yekun Markdown faylını DOCX formatına çeviririk.

```bash
# Pandoc əmri
pandoc /home/ubuntu/50_gunde_si/kitab_yekun.md \
    -o /home/ubuntu/50_gunde_si/50_Gunde_Sun_i_Intellekt.docx \
    --toc \
    --toc-depth=2 \
    --standalone \
    --metadata title="50 Gündə Süni-İntellekt" \
    --metadata author="Manus"
```

**Kodun İzahı:**
*   `-o`: Çıxış faylının adını təyin edir.
*   `--toc`: **Məzmun Cədvəli** (Table of Contents) yaradır.
*   `--toc-depth=2`: Məzmun cədvəlinə yalnız H1 və H2 başlıqlarını daxil edir.
*   `--standalone`: Tam, müstəqil bir DOCX faylı yaradır.
*   `--metadata`: Kitabın metadata məlumatlarını (başlıq, müəllif) əlavə edir.

### 3. Təhvil

Bütün proses tamamlandı. Yekun DOCX faylı sizə təhvil verilməyə hazırdır.

**Təbrik edirəm!** Siz bu kitabı yazmaqla Süni İntellekt sahəsində böyük bir addım atdınız.

### 💡 Günün Tapşırığı: Təhvil

Bu günün tapşırığı, bu kitabın bütün biliklərini tətbiq etmək və öz LLM-inizi yaratmaqdır!

**Səyahətiniz uğurlu olsun!** 👋

***

**Söz Sayı:** 750 söz.
