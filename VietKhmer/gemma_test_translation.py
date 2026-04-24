import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN VÀ THAM SỐ
# ==========================================
# Đường dẫn tới thư mục model ĐÃ ĐƯỢC MERGE ở bước trước
MODEL_DIR = "/mnt/e/AILAB/MODELS/gemma-2b-vi-km-final" 

# Tham số Generation (Cực kỳ quan trọng cho dịch thuật)
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.1      # Để thấp (0.1 - 0.3) để bản dịch bám sát bản gốc, không sáng tạo lung tung
TOP_P = 0.9
REPETITION_PENALTY = 1.1 # Phạt lặp từ (Giúp tránh lỗi mô hình 2B hay lặp lại một từ Khmer nhiều lần)

# ==========================================
# 2. TẢI MÔ HÌNH VÀ TOKENIZER
# ==========================================
print(f"🔄 Đang tải mô hình từ: {MODEL_DIR}")
start_time = time.time()

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
# Load ở chế độ bfloat16/float16 để suy luận nhanh và tiết kiệm VRAM
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.bfloat16,
    device_map="auto" # Tự động đưa lên GPU nếu có
)
model.eval() # Chuyển sang chế độ suy luận

print(f"✅ Tải mô hình thành công! (Mất {time.time() - start_time:.2f} giây)\n")

# ==========================================
# 3. HÀM TẠO PROMPT VÀ DỊCH
# ==========================================
def translate(text: str, direction: str) -> str:
    """
    direction: 'vi2km' hoặc 'km2vi'
    """
    # BẮT BUỘC: Prompt phải giống hệt 100% lúc Training
    if direction == 'vi2km':
        prompt = f"Instruction: Translate Vietnamese to Khmer.\nInput: {text}\nOutput: "
    elif direction == 'km2vi':
        prompt = f"Instruction: Translate Khmer to Vietnamese.\nInput: {text}\nOutput: "
    else:
        return "Hướng dịch không hợp lệ!"

    # Chuyển đổi prompt thành tensor
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Sinh chuỗi (Generate)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    # Giải mã kết quả (chỉ lấy phần sau phần prompt đầu vào)
    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[0][input_length:]
    
    translation = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return translation

# ==========================================
# 4. CHẾ ĐỘ 1: DỊCH TƯƠNG TÁC (INTERACTIVE)
# ==========================================
def interactive_mode():
    print("="*50)
    print("🤖 CHẾ ĐỘ DỊCH TƯƠNG TÁC (Gõ 'exit' để thoát)")
    print("="*50)
    
    while True:
        mode = input("\nChọn hướng dịch (1: VI -> KM, 2: KM -> VI, exit: Thoát): ").strip()
        if mode.lower() == 'exit':
            break
        
        if mode not in ['1', '2']:
            print("Vui lòng chọn 1 hoặc 2.")
            continue
            
        direction = 'vi2km' if mode == '1' else 'km2vi'
        lang_name = "Tiếng Việt" if mode == '1' else "Tiếng Khmer"
        
        text_input = input(f"Nhập câu {lang_name}: ").strip()
        if not text_input:
            continue
            
        print("⏳ Đang dịch...")
        result = translate(text_input, direction)
        print(f"🎯 Kết quả: {result}")

# ==========================================
# 5. CHẾ ĐỘ 2: DỊCH HÀNG LOẠT TỪ FILE
# ==========================================
def file_mode(input_file: str, output_file: str, direction: str):
    print(f"\n🔄 Đang dịch file: {input_file} -> {output_file}")
    
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        lines = fin.readlines()
        for i, line in enumerate(lines):
            text = line.strip()
            if text:
                result = translate(text, direction)
                fout.write(result + '\n')
                print(f"Đã dịch {i+1}/{len(lines)} câu.")
                
    print("✅ Đã dịch xong file!")

# ==========================================
# THỰC THI
# ==========================================
if __name__ == "__main__":
    # Chọn chế độ bạn muốn chạy
    
    # 1. Chạy chế độ gõ trực tiếp
    interactive_mode()
    
    # 2. Hoặc bỏ comment 2 dòng dưới đây để dịch cả 1 file test
    # file_mode("test_vi.txt", "predict_km.txt", direction="vi2km")