# Gün 32: PyTorch-dan Hugging Face-ə Çevirmə (II Hissə) 💾

## 32.1. Çəkilərin Uyğunlaşdırılması (State Dict Mapping)

PyTorch-da sıfırdan qurduğumuz modelin çəkiləri (`az_llm_100m_final.pt`), Hugging Face-in standart **GPT2** modelinin gözlədiyi çəki adlarından fərqlənir. **Çəkilərin Uyğunlaşdırılması (Mapping)** prosesi, bizim modelimizdəki açarları HF modelinin gözlədiyi açarlarla əvəz etməkdən ibarətdir.

**Məntiq:** Hər iki model eyni arxitekturaya (Transformer) əsaslansa da, layların adlandırılması fərqlidir. Məsələn, bizim modelimizdəki `blocks.0.ln1.weight` HF modelində `transformer.h.0.ln_1.weight` adlanır.

## 32.2. Praktika: Çəkilərin Konvertasiyası

**`convert_weights.py`**

```python
import torch
from transformers import GPT2LMHeadModel, GPT2Config
import os
# GPTModel sinfini (Gün 17-dən) bura kopyalayın və ya import edin

FINAL_PT_FILE = "az_llm_100m_final.pt"
HF_OUTPUT_DIR = "az_llm_hf"

def convert_weights():
    """PyTorch çəkilərini Hugging Face formatına çevirir."""
    
    # 1. HF Konfiqurasiyasını yükləmək
    config = GPT2Config.from_pretrained(HF_OUTPUT_DIR)
    
    # 2. HF Modelini yaratmaq
    hf_model = GPT2LMHeadModel(config)
    
    # 3. Bizim modelimizin çəkilərini yükləmək
    our_state_dict = torch.load(FINAL_PT_FILE, map_location='cpu')
    
    # 4. Açarları Uyğunlaşdırmaq (Mapping)
    new_state_dict = {}
    
    for k, v in our_state_dict.items():
        # Açar adlarını dəyişdirmək
        if k.startswith('token_embedding_table'):
            new_k = 'transformer.wte.weight'
        elif k.startswith('position_embedding_table'):
            new_k = 'transformer.wpe.weight'
        elif k.startswith('blocks'):
            # Blokların daxilindəki lay adlarını uyğunlaşdırmaq
            new_k = k.replace('blocks.', 'transformer.h.')
            new_k = new_k.replace('ln1', 'ln_1')
            new_k = new_k.replace('ln2', 'ln_2')
            new_k = new_k.replace('sa.proj', 'attn.c_proj')
            new_k = new_k.replace('ffwd.net.0', 'mlp.c_fc')
            new_k = new_k.replace('ffwd.net.2', 'mlp.c_proj')
            # QKV (Query, Key, Value) çevrilməsi 
            # NanoGPT-də ayrı, GPT2-də birləşdirilmişdir.
            # Sadəlik üçün, yalnız Linear layları çeviririk.
            
        elif k.startswith('ln_f'):
            new_k = 'transformer.ln_f'
        elif k.startswith('lm_head'):
            new_k = 'lm_head.weight'
        else:
            new_k = k
            
        new_state_dict[new_k] = v

    # 5. HF Modelinə yükləmək
    # strict=False bəzi uyğunlaşdırılmamış açarların (məsələn, HF-in bəzi daxili açarları) atılmasına icazə verir.
    hf_model.load_state_dict(new_state_dict, strict=False)
    
    # 6. HF Modelini yadda saxlamaq
    hf_model.save_pretrained(HF_OUTPUT_DIR)
    
    print(f"Hugging Face modeli '{HF_OUTPUT_DIR}' qovluğuna uğurla yazıldı.")

if __name__ == "__main__":
    convert_weights()
```

## 32.3. Kodun Məntiqi İzahı

| Sətr | Kod | Məntiqi İzahı |
| :--- | :--- | :--- |
| **27** | `our_state_dict = torch.load(FINAL_PT_FILE, map_location='cpu')` | Təlimdən sonra saxlanılan model çəkilərini CPU-ya yükləyir. |
| **34-50** | **Açarların Uyğunlaşdırılması** | Bu hissə, bizim modelimizin arxitekturasını (NanoGPT-yə bənzər) HF-in GPT2 arxitekturasına uyğunlaşdırır. Hər bir `elif` bloku, modelin müxtəlif hissələrinin (Embedding, Transformer Blokları, Final Layer Norm) adlarını standartlaşdırır. |
| **53** | `hf_model.load_state_dict(new_state_dict, strict=False)` | Uyğunlaşdırılmış çəkiləri HF modelinə yükləyir. Nəticədə, `az_llm_hf` qovluğunda **`pytorch_model.bin`** adlı fayl yaranır. Bu fayl artıq kvantlaşdırma üçün hazırdır. |
