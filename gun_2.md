# 📚 50 Gündə Süni-İntellekt: Gün 2

## Python: Sıfırdan Başlanğıc 🐍

Salam! Dünən Süni İntellektin əsaslarını öyrəndik. Bu gün isə LLM yaratmaq üçün istifadə edəcəyimiz **əsas aləti** – **Python** proqramlaşdırma dilini öyrənməyə başlayırıq.

Unutmayın, siz Python bilmədiyinizi qeyd etdiniz. Buna görə də, hər şeyi ən sadə dildə, **Windows** əməliyyat sistemi üçün addım-addım izah edəcəyəm.

### 1. Niyə Məhz Python?

Dünyada yüzlərlə proqramlaşdırma dili var. Bəs niyə Süni İntellekt və Dərin Öyrənmə layihələrinin demək olar ki, hamısı **Python** üzərində qurulur?

*   **Sadəlik:** Python çox sadə və oxunaqlı bir dildir. Sanki ingilis dilində cümlələr yazırsınız. Bu, yeni başlayanlar üçün ideal seçimdir.
*   **Nəhəng Kitabxana Ekosistemi:** Python-un Süni İntellekt üçün hazırlanmış **PyTorch**, **TensorFlow**, **Hugging Face** kimi nəhəng kitabxanaları var. Bizim LLM layihəmizdə bu kitabxanalardan istifadə edəcəyik.
*   **İcma Dəstəyi:** Python-un çox böyük bir istifadəçi icması var. Hər hansı bir problem qarşısında qalsanız, cavabı internetdə asanlıqla tapa bilərsiniz.

### 2. Windows-da Python-un Quraşdırılması

**Diqqət:** Bu addımlar sizin şəxsi kompüterinizdə (Windows) icra edilməlidir.

#### Addım 1: Python-u Yükləmək

1.  Brauzerinizi açın və **Python.org** rəsmi saytına daxil olun.
2.  "Downloads" (Yükləmələr) bölməsinə keçin.
3.  Windows üçün ən son stabil versiyanı (məsələn, Python 3.11.x) yükləyin.

#### Addım 2: Quraşdırma Prosesi

Yüklədiyiniz faylı (məsələn, `python-3.11.x.exe`) iki dəfə klikləyin. Quraşdırma pəncərəsi açılacaq.

⚠️ **Çox Vacib Məqam:** Pəncərənin ən aşağısında yerləşən **"Add python.exe to PATH"** (Python.exe-ni PATH-a əlavə et) qutusunu **MÜTLƏQ** işarələyin! Bu, Python əmrlərini kompüterinizin istənilən yerindən işlədə bilməyiniz üçün vacibdir.

1.  "Add python.exe to PATH" qutusunu işarələyin.
2.  "Install Now" (İndi Quraşdır) düyməsini sıxın.

Quraşdırma bir neçə dəqiqə çəkəcək.

#### Addım 3: Quraşdırmanın Yoxlanılması

Quraşdırma bitdikdən sonra, Python-un düzgün işlədiyini yoxlayaq:

1.  Windows axtarış çubuğuna **"CMD"** yazın və **Command Prompt** (Əmr Sətiri) proqramını açın.
2.  Açılan qara pəncərədə aşağıdakı əmri yazın və **Enter** düyməsini basın:

```bash
python --version
```

**İzah:**
*   `python`: Python proqramını çağıran əsas əmrdir.
*   `--version`: Python-dan quraşdırılmış versiya nömrəsini göstərməsini istəyirik.

Nəticə olaraq, quraşdırdığınız versiyanı görməlisiniz (məsələn, `Python 3.11.5`). Əgər bu nömrəni görürsünüzsə, **təbriklər!** Python uğurla quraşdırılıb.

### 3. Paket İdarəetmə Sistemi: PIP

Python-un gücü onun **paketlərində** (kitabxanalarında) gizlidir. Bizim LLM layihəmizdə istifadə edəcəyimiz PyTorch, Hugging Face kimi paketləri kompüterimizə yükləmək üçün **PIP** adlı bir alətdən istifadə edəcəyik.

> **PIP** (Package Installer for Python) — Python paketlərini quraşdırmaq, yeniləmək və silmək üçün istifadə olunan standart idarəetmə sistemidir.

Python-u quraşdırarkən "Add python.exe to PATH" qutusunu işarələdinizsə, PIP də avtomatik quraşdırılıb. Yoxlayaq:

```bash
pip --version
```

Nəticə olaraq, PIP-in versiyasını görməlisiniz (məsələn, `pip 23.2.1 from ...`).

### 4. İlk Python Kodumuz: "Salam Dünya!"

İndi isə ilk Python kodumuzu yazaq.

1.  Command Prompt-u açın.
2.  Aşağıdakı əmri yazın və **Enter** düyməsini basın:

```bash
python
```

Bu əmr sizi **Python İnteraktiv Mühitinə** daxil edəcək (üç dənə `>>>` işarəsi görünəcək).

3.  İndi isə kodumuzu yazaq:

```python
print("Salam, 50 Gündə Süni-İntellekt!")
```

**İzah:**
*   `print()`: Python-da ekrana məlumat çıxarmaq üçün istifadə olunan əsas funksiyadır.
*   `"Salam, 50 Gündə Süni-İntellekt!"`: Ekrana çıxarılmasını istədiyimiz mətndir. Mətn həmişə dırnaq işarələri (`""`) arasında yazılır.

Nəticə olaraq, ekranda **Salam, 50 Gündə Süni-İntellekt!** yazısını görəcəksiniz.

İnteraktiv mühitdən çıxmaq üçün `exit()` yazıb Enter-ə basın.

### 💡 Günün Tapşırığı: Praktika

1.  Python-u quraşdırın və `python --version` əmri ilə yoxlayın.
2.  PIP-in quraşdırıldığını `pip --version` əmri ilə yoxlayın.
3.  Python interaktiv mühitində öz adınızı ekrana çıxaran bir `print()` əmri yazın.

**Sabah görüşənədək!** 👋 Sabah **virtual iş mühitini** necə quracağımızı öyrənəcəyik. Bu, layihələrimizi səliqəli saxlamaq üçün çox vacibdir.

***

**Söz Sayı:** 650 söz.
