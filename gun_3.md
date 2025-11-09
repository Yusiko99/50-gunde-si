# 📚 50 Gündə Süni-İntellekt: Gün 3

## İş Mühitinin Qurulması: Səliqəli Başlanğıc 🛠️

Salam, gələcəyin Süni İntellekt mütəxəssisi!

Dünən Python-u uğurla quraşdırdıq. Bu gün isə ən az Python-un özü qədər vacib olan bir mövzuya – **İş Mühitinin Qurulmasına** toxunacağıq. Bu addım, layihələrimizin bir-birinə qarışmaması və səliqəli qalması üçün təməl daşıdır.

### 1. Niyə İş Mühiti Lazımdır?

Təsəvvür edin ki, sizin iki fərqli layihəniz var:
1.  **Layihə A:** Köhnə bir kitabxana versiyası (`PyTorch 1.0`) tələb edir.
2.  **Layihə B (Bizim LLM):** Ən yeni kitabxana versiyası (`PyTorch 2.0`) tələb edir.

Əgər siz bu iki layihəni eyni kompüter mühitində işlətsəniz, birinin tələb etdiyi köhnə versiyanı quraşdırdıqda, digərinin tələb etdiyi yeni versiya pozulacaq. Bu, **"Dependency Hell"** (Asılılıq Cəhənnəmi) adlanır.

Bu problemi həll etmək üçün **Virtual Mühitlərdən** istifadə edirik.

> **Virtual Mühit (Virtual Environment)** — hər bir layihə üçün özünəməxsus, təcrid olunmuş bir qutu (sandbox) yaratmaq deməkdir. Bu qutunun içində quraşdırdığınız proqramlar və kitabxanalar, kompüterinizin əsas sisteminə və ya digər layihələrə təsir etmir.

### 2. Anaconda/Miniconda: Ən Güclü Alət

Python layihələrində virtual mühit yaratmaq üçün bir neçə alət var (`venv`, `virtualenv`). Lakin, Süni İntellekt və Dərin Öyrənmə layihələrində ən çox istifadə olunan və ən güclüsü **Conda**-dır.

*   **Anaconda:** Elmi hesablamalar üçün lazım olan yüzlərlə paketi (Python, R, Spyder, Jupyter və s.) özündə birləşdirən böyük bir proqram paketidir.
*   **Miniconda:** Yalnız **Conda** idarəetmə sistemini və Python-u ehtiva edən, daha yüngül versiyadır. Bizim LLM layihəmiz üçün **Miniconda** daha ideal seçimdir, çünki lazım olmayan paketlərlə kompüterimizi yükləməyəcəyik.

#### Addım 1: Miniconda-nın Quraşdırılması

1.  Brauzerinizi açın və **Miniconda** axtarışını edin və ya rəsmi saytına daxil olun.
2.  Windows üçün olan **Python 3.x** versiyasını yükləyin.
3.  Yüklədiyiniz faylı iki dəfə klikləyin və quraşdırma prosesini standart olaraq davam etdirin. Quraşdırma zamanı "Add Anaconda to my PATH environment variable" seçimini **işarələməyin** (bu, Anaconda-nın özü üçün tövsiyə olunur, lakin Miniconda-da bəzən problemlər yarada bilər). Sadəcə "Just Me" (Yalnız Mən) seçimi ilə irəliləyin.

#### Addım 2: Conda-nın Yoxlanılması

Quraşdırma bitdikdən sonra, Windows axtarış çubuğuna **"Anaconda Prompt"** yazın və açın. Bütün Conda əmrlərini bu pəncərədə icra edəcəyik.

Aşağıdakı əmri yazın və **Enter** düyməsini basın:

```bash
conda --version
```

Nəticə olaraq, Conda-nın versiyasını görməlisiniz (məsələn, `conda 23.7.4`).

### 3. Virtual Mühitin Yaradılması (Praktika)

İndi isə LLM layihəmiz üçün xüsusi bir virtual mühit yaradaq. Adını **`llm_50gun`** qoyacağıq.

```bash
conda create --name llm_50gun python=3.11
```

**İzah:**
*   `conda create`: Conda-ya yeni bir virtual mühit yaratmasını əmr edir.
*   `--name llm_50gun`: Yaratdığımız mühitə **`llm_50gun`** adını veririk.
*   `python=3.11`: Bu mühitin içində **Python 3.11** versiyasının quraşdırılmasını tələb edirik.

Əmr icra olunduqdan sonra, Conda sizdən təsdiq istəyəcək (`[y/n]`). **`y`** yazıb Enter-ə basın.

### 4. Virtual Mühitin Aktivləşdirilməsi

Mühiti yaratdıq, indi onu **aktivləşdirməliyik**.

```bash
conda activate llm_50gun
```

**İzah:**
*   `conda activate`: Yaratdığımız virtual mühiti işə salır.

Əmr icra olunduqdan sonra, **Anaconda Prompt** pəncərənizin əvvəlində `(base)` əvəzinə **`(llm_50gun)`** yazıldığını görəcəksiniz. Bu o deməkdir ki, siz artıq LLM layihənizin təcrid olunmuş, təmiz qutusunun içindəsiniz!

Bundan sonra quraşdıracağımız bütün kitabxanalar (PyTorch, Hugging Face və s.) yalnız bu `llm_50gun` mühitinin içində olacaq.

### 5. Virtual Mühitdən Çıxış

İşiniz bitdikdə mühitdən çıxmaq üçün:

```bash
conda deactivate
```

Pəncərənin əvvəlindəki ad yenidən `(base)` olacaq.

### 💡 Günün Tapşırığı: Praktika

1.  Miniconda-nı quraşdırın.
2.  **`llm_50gun`** adlı virtual mühit yaradın.
3.  Mühiti aktivləşdirin (`conda activate llm_50gun`).
4.  Mühitin içində Python versiyasını yoxlayın (`python --version`).
5.  Mühitdən çıxın (`conda deactivate`).

**Sabah görüşənədək!** 👋 Sabah ən vacib addımlardan birini atacağıq: **GPU Sürətləndirilməsi** üçün **CUDA** və **PyTorch**-u necə quraşdıracağımızı öyrənəcəyik. Bu, modelimizin təlim sürətini yüzlərlə dəfə artıracaq!

***

**Söz Sayı:** 680 söz.
