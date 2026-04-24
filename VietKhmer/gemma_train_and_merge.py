#accelerate launch train_and_merge.py


import os
import gc
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer, SFTConfig # Import SFTConfig thay cho TrainingArguments
from accelerate import PartialState

# ==========================================
# 1. CẤU HÌNH THÔNG SỐ VÀ MÔI TRƯỜNG DDP
# ==========================================
# PartialState giúp quản lý các luồng (process) khi chạy đa GPU
state = PartialState()
device_map = {'': state.process_index} # Gán chính xác từng tiến trình vào từng GPU

MODEL_ID = "google/gemma-2-2b-it"
ADAPTER_DIR = "/mnt/e/AILAB/MODELS/gemma-2b-vi-km-adapter"
FINAL_DIR = "/mnt/e/AILAB/MODELS/gemma-2b-vi-km-final" # Đã sửa lỗi dư dấu '/' trong đường dẫn
CSV_FILE_PATH = "/mnt/e/AILAB/DATASET_TEXT_TRAIN/data_vi_kh.csv"

#hf_token = '' 

# Tối ưu cho GPU 12GB VRAM x 2
BATCH_SIZE = 2 
GRAD_ACCUMULATION = 4 
LEARNING_RATE = 1e-5
EPOCHS = 10
MAX_SEQ_LENGTH = 512

# ==========================================
# 2. ĐỌC VÀ XỬ LÝ DỮ LIỆU TỪ CSV
# ==========================================
if state.is_main_process:
    print(f"🔄 Đang đọc dữ liệu từ file CSV: {CSV_FILE_PATH}...")

dataset = load_dataset('csv', data_files={'train': CSV_FILE_PATH})['train']

def format_prompts(example):
    vi_text = example['ViContent']
    km_text = example['KhContent']
    
    prompt_vi_km = f"Instruction: Translate Vietnamese to Khmer.\nInput: {vi_text}\nOutput: {km_text}"
    prompt_km_vi = f"Instruction: Translate Khmer to Vietnamese.\nInput: {km_text}\nOutput: {vi_text}"
    
    return {"text": [prompt_vi_km, prompt_km_vi]}

formatted_dataset = dataset.map(
    format_prompts, 
    batched=True, 
    remove_columns=dataset.column_names
)

if state.is_main_process:
    print(f"✅ Độ dài dữ liệu ban đầu: {len(dataset)}")

formatted_dataset = formatted_dataset.shuffle(seed=42)

if state.is_main_process:    
    print(f"✅ Đã chuẩn bị xong {len(formatted_dataset)} mẫu dữ liệu huấn luyện (Đã nhân đôi do tạo prompt 2 chiều).")

# ==========================================
# 3. TẢI MÔ HÌNH VÀ CẤU HÌNH QLoRA
# ==========================================
if state.is_main_process:
    print("\n🔄 Đang tải Model và cấu hình 4-bit...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" 

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map=device_map,
    token=hf_token
)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=64, 
    lora_alpha=32,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

#model = get_peft_model(model, lora_config)

# ==========================================
# 4. BẮT ĐẦU HUẤN LUYỆN (ĐÃ CẬP NHẬT TRánh LỖI TRL)
# ==========================================
if state.is_main_process:
    print("\n🚀 Bắt đầu quá trình huấn luyện Đa GPU...")

# Sử dụng SFTConfig tích hợp toàn bộ các tham số của Trainer
sft_config = SFTConfig(
    output_dir=ADAPTER_DIR,
    dataset_text_field="text",        # FIX 1: Đưa tên cột dữ liệu vào đây
    max_length=MAX_SEQ_LENGTH,    # FIX 2: Đưa chiều dài chuỗi vào đây
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUMULATION,
    learning_rate=LEARNING_RATE,
    num_train_epochs=EPOCHS,
    logging_steps=10,
    save_strategy="epoch",
    optim="paged_adamw_32bit",
    bf16=True, 
    max_grad_norm=0.3,
    warmup_steps=100,                 # FIX 3: Dùng warmup_steps thay cho warmup_ratio
    lr_scheduler_type="cosine",
    ddp_find_unused_parameters=False, 
    report_to="none",
    packing=False                     # Tắt packing mặc định để tránh cảnh báo
)

# Khởi tạo Trainer gọn gàng, không truyền tham số SFT dư thừa bên ngoài nữa
trainer = SFTTrainer(
    model=model,
    train_dataset=formatted_dataset,
    args=sft_config,                  # Đã gom hết cấu hình vào SFTConfig
    processing_class=tokenizer,
    peft_config=lora_config,
)

trainer.train()

if state.is_main_process:
    trainer.model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    print(f"\n✅ Đã lưu LoRA adapter thành công tại: {ADAPTER_DIR}")

# ==========================================
# 5. DỌN DẸP BỘ NHỚ VÀ HỢP NHẤT MÔ HÌNH (MERGE)
# ==========================================
state.wait_for_everyone()

if state.is_main_process:
    print("\n🧹 Đang dọn dẹp VRAM để chuẩn bị Merge Model...")
    
    del trainer
    del model
    
    gc.collect()
    torch.cuda.empty_cache()

    print("\n🔄 Đang load Base Model (bfloat16) để Merge...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token
    )

    print("🔄 Đang gộp Adapter vào Base Model...")
    merged_model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    merged_model = merged_model.merge_and_unload()

    print(f"💾 Đang lưu mô hình hoàn chỉnh vào: {FINAL_DIR}")
    merged_model.save_pretrained(FINAL_DIR)
    tokenizer.save_pretrained(FINAL_DIR)
    
    print("\n🎉 HOÀN TẤT TẤT CẢ! Mô hình của bạn đã sẵn sàng sử dụng.")