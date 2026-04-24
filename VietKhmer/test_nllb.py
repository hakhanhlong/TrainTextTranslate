import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel, PeftConfig

MODEL_ID = "facebook/nllb-200-1.3B"
ADAPTER_PATH = "/mnt/e/AILAB/MODELS/nllb-vi-km-lora/final_adapter"
SRC_LANG = "vie_Latn"
TGT_LANG = "khm_Khmr"

# 1. Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, src_lang=SRC_LANG)

# 2. Load Base Model (Có thể load 8-bit hoặc fp16 tùy phần cứng lúc chạy)
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.float16, 
    device_map="auto"
)

# 3. Gắn LoRA Adapter vào Base Model
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

# 4. Viết hàm dịch
def translate_vi_to_km(text):
    # 1. Tokenize văn bản đầu vào
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # 2. Lấy ID của ngôn ngữ đích (FIX LỖI TẠI ĐÂY)
    # NLLB coi các mã ngôn ngữ là special tokens
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(TGT_LANG)
    
    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=128,
            num_beams=5
        )
        
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

# Chạy thử
text_vi = "Bản làng ở Lai Châu đang trở thành điểm đến du lịch hấp dẫn, mang lại lợi ích lâu dài cho cả cộng đồng và du khách."
print("Khmer translation:", translate_vi_to_km(text_vi))