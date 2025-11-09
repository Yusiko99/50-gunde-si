# 📚 50 Gündə Süni-İntellekt: Gün 32

## PyTorch-dan Hugging Face-ə Çevirmə (II Hissə) 🧩

Salam! Dünən PyTorch çəkilərini Hugging Face (HF) modelinə köçürmə prosesinə başladıq. Bu gün isə ən çətin hissəni – **Çoxbaşlı Diqqət (MHA)** qatının çəkilərinin uyğunlaşdırılmasını və **Tokenizatorun** saxlanmasını tamamlayırıq.

### 1. MHA Çəkilərinin Köçürülməsi

Bizim NanoGPT modelimizdə Q, K, V (Query, Key, Value) çəkiləri **`c_attn`** adlı tək bir xətti qatda birləşdirilmişdi. Hugging Face GPT-2 modelində isə bu çəkilər ayrı-ayrı qatlarda saxlanılır.

Bizim `c_attn` çəkisini 3 bərabər hissəyə bölüb, HF modelinin Q, K, V çəkilərinə kopyalamalıyıq.

#### `export_hf.py` Skriptinin Yenilənməsi

`convert_nano_to_hf` funksiyasına bu hissəni əlavə edirik:

```python
# export_hf.py (convert_nano_to_hf funksiyasının içində)

# ... (əvvəlki kodlar) ...

    # Transformer Bloklarının (12 ədəd) çəkilərini köçürürük
    for i in range(nano_model.config.n_layer):
        # ... (Layer Norms və FFN-in köçürülməsi) ...
        
        # Multi-Head Attention (MHA) Çəkilərinin Köçürülməsi
        
        # 1. NanoGPT-dən birləşdirilmiş QKV çəkilərini alırıq
        qkv_weight = nano_state_dict[f'transformer.h.{i}.attn.c_attn.weight']
        qkv_bias = nano_state_dict[f'transformer.h.{i}.attn.c_attn.bias']
        
        # 2. Çəkiləri 3 bərabər hissəyə bölürük (Q, K, V)
        # Hər hissənin ölçüsü n_embd (768)
        q_w, k_w, v_w = torch.chunk(qkv_weight, 3, dim=0)
        q_b, k_b, v_b = torch.chunk(qkv_bias, 3, dim=0)
        
        # 3. Hugging Face modelinə kopyalayırıq
        # HF-də Q, K, V birləşdirilmiş şəkildə saxlanılır
        hf_qkv_weight = torch.cat([q_w, k_w, v_w], dim=0)
        hf_qkv_bias = torch.cat([q_b, k_b, v_b], dim=0)
        
        # HF modelinin state dict-inə kopyalayırıq
        hf_state_dict[f'transformer.h.{i}.attn.c_attn.weight'].copy_(hf_qkv_weight)
        hf_state_dict[f'transformer.h.{i}.attn.c_attn.bias'].copy_(hf_qkv_bias)
        
        # MHA-nın son proyeksiya qatını kopyalayırıq
        hf_state_dict[f'transformer.h.{i}.attn.c_proj.weight'].copy_(
            nano_state_dict[f'transformer.h.{i}.attn.c_proj.weight']
        )
        hf_state_dict[f'transformer.h.{i}.attn.c_proj.bias'].copy_(
            nano_state_dict[f'transformer.h.{i}.attn.c_proj.bias']
        )
        
        # ... (qalan kodlar) ...
```

**Kodun İzahı:**
*   `torch.chunk(qkv_weight, 3, dim=0)`: Birləşdirilmiş çəki matrisini (ölçüsü 3 * 768) 3 bərabər hissəyə (hər biri 768 ölçülü) bölür.
*   `torch.cat([q_w, k_w, v_w], dim=0)`: Bəzi HF modelləri Q, K, V-ni birləşdirilmiş şəkildə saxlayır. Biz də bölüb yenidən birləşdiririk.

### 2. Tokenizatorun Saxlanması

Bizim tokenizatorumuz Hugging Face-in `tokenizers` kitabxanası ilə yaradılıb. Onu HF-in `transformers` kitabxanasının istifadə edə biləcəyi formata çevirməliyik.

#### `export_hf.py` Skriptinin Yenilənməsi (Əsas İcra Bloku)

```python
# export_hf.py (Əsas İcra Bloku)

# ... (əvvəlki kodlar) ...

    # 5. Tokenizatoru saxla
    tokenizer = Tokenizer.from_file("az_bpe_tokenizer.json")
    
    # Hugging Face Tokenizatorunu yaratmaq üçün
    from transformers import PreTrainedTokenizerFast
    
    # Tokenizatoru HF formatında saxlamaq üçün
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<|endoftext|>", # Başlanğıc tokeni
        eos_token="<|endoftext|>", # Son tokeni
        unk_token="<|endoftext|>", # Bilinməyən token
        pad_token="<|endoftext|>", # Padding tokeni
    )
    
    # Tokenizatoru qovluğa yazırıq
    hf_tokenizer.save_pretrained("az_llm_hf")
    print("Tokenizator 'az_llm_hf' qovluğuna yazıldı.")
    
    # 6. Modeli saxla
    hf_model.save_pretrained("az_llm_hf")
    print("Model 'az_llm_hf' qovluğuna yazıldı.")
    
    print("\nPyTorch-dan Hugging Face-ə çevirmə uğurla tamamlandı!")
```

### 3. Yekun İcra

İndi `export_hf.py` skriptini icra etdikdə, **`az_llm_hf`** adlı bir qovluq yaranacaq. Bu qovluğun içində modelin çəkiləri (`pytorch_model.bin`) və tokenizator faylları (`tokenizer.json`, `tokenizer_config.json` və s.) olacaq.

Bu qovluq artıq Hugging Face ekosistemində istifadə oluna bilər.

### 💡 Günün Tapşırığı: Praktika

1.  `export_hf.py` faylını yuxarıdakı kodla tamamlayın.
2.  `pip install transformers` əmrini icra edin (əgər əvvəlki gün etməmisinizsə).
3.  Skripti icra edin: `python export_hf.py`.
4.  Yaranan `az_llm_hf` qovluğunun içindəki faylları yoxlayın.

**Sabah görüşənədək!** 👋 Sabah **GGUF Formatına Çevirmə** prosesinə başlayırıq.

***

**Söz Sayı:** 750 söz.
