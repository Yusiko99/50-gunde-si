# Gün 50: DOCX-ə Çevrilmə və Təhvil 🎁

## 50.1. Yekun Markdown Faylının Qurulması

Bu gün, 50 günlük təlim materialının son formatda təqdim edilməsi prosesi tamamlanır. Bütün məzmun hissələri (Ön Söz, Məzmun Cədvəli, Əsas Mətn, Yekun Söz) ardıcıl olaraq bir yekun Markdown faylında birləşdirilir.

**Birləşdirmə Ardıcıllığı:**

1.  Gün 48 (Ön Söz və Məzmun Cədvəli)
2.  Gün 1-dən Gün 47-yə qədər olan əsas mətn
3.  Gün 49 (Yekun Söz)

## 50.2. DOCX Formatına Çevrilmə

Təlim materialının tələb olunan **DOCX** formatında təqdim edilməsi üçün **Pandoc** alətindən istifadə olunur.

**Pandoc Əmrlərinin Məntiqi:**

```bash
# 1. Bütün hissələri birləşdirmək
cat /home/ubuntu/50_gunde_si/gun_48_obj.md \
    /home/ubuntu/50_gunde_si/kitab_esas_metn_obj.md \
    /home/ubuntu/50_gunde_si/gun_49_obj.md \
    > /home/ubuntu/50_gunde_si/50_Gunde_Sun_i_Intellekt_Yekun_Obj.md

# 2. Markdown-u DOCX-ə çevirmək
pandoc /home/ubuntu/50_gunde_si/50_Gunde_Sun_i_Intellekt_Yekun_Obj.md \
    -o /home/ubuntu/50_gunde_si/50_Gunde_Sun_i_Intellekt_Obyektiv.docx \
    --toc \
    --toc-depth=2 \
    --standalone \
    --wrap=none \
    --metadata title="50 Gündə Süni-İntellekt: Azərbaycan Dilində LLM-i Sıfırdan Qurmaq" \
    --metadata author="Manus AI"
```

| Pandoc Parametri | Məntiqi Əsas |
| :--- | :--- |
| **`-o`** | Çıxış faylının adını və formatını təyin edir. |
| **`--toc`** | Avtomatik olaraq başlıqlara əsaslanan **Məzmun Cədvəli** yaradır. |
| **`--toc-depth=2`** | Məzmun Cədvəlinə yalnız 2-ci səviyyəyə qədər başlıqları daxil edir. |
| **`--standalone`** | Tam, müstəqil bir DOCX faylı yaradır. |
| **`--wrap=none`** | Kod bloklarının sətirlərinin bükülməsinin qarşısını alır. |

## 50.3. Təhvil

Bu prosesin sonunda, obyektiv və sistemin məntiqinə fokuslanmış təlim materialı yekun DOCX formatında təqdim edilir.
