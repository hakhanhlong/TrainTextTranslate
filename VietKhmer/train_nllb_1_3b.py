import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    BitsAndBytesConfig
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ==========================================
# 1. CẤU HÌNH CƠ BẢN
# ==========================================
MODEL_ID = "facebook/nllb-200-1.3B" # Đổi thành "facebook/nllb-200-3.3B" nếu GPU > 24GB VRAM
DATA_PATH = "/mnt/e/AILAB/DATASET_TEXT_TRAIN/data_vi_kh.csv"
OUTPUT_DIR = "/mnt/e/AILAB/MODELS/nllb-vi-km-lora_1m"

SRC_LANG = "vie_Latn"
TGT_LANG = "khm_Khmr"
MAX_LENGTH = 256

# ==========================================
# 2. LOAD DỮ LIỆU TỪ CSV
# ==========================================
print("Loading dataset...")
# Tải file csv và tự động chia tập train (90%) và validation (10%)


dataset = load_dataset("csv", data_files=DATA_PATH)
dataset = dataset["train"].train_test_split(test_size=0.01, seed=42)

# ==========================================
# 3. TOKENIZER & PREPROCESSING
# ==========================================
print("Setting up tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, src_lang=SRC_LANG)

def preprocess_function(examples):
    inputs = [str(ex) for ex in examples["ViContent"]]
    targets = [str(ex) for ex in examples["KhContent"]]
    
    # Thiết lập ngôn ngữ nguồn
    tokenizer.src_lang = SRC_LANG
    
    # Tokenize cả input và target cùng lúc bằng tham số text_target
    # Đây là cách làm chuẩn hiện nay, thay thế cho as_target_tokenizer()
    model_inputs = tokenizer(
        inputs, 
        text_target=targets, 
        max_length=MAX_LENGTH, 
        truncation=True
    )
    
    return model_inputs

print("Tokenizing datasets...")
tokenized_datasets = dataset.map(
    preprocess_function, 
    batched=True, 
    remove_columns=dataset["train"].column_names,
    num_proc=4 # Sử dụng 4 luồng CPU để xử lý nhanh hơn
)

# ==========================================
# 4. LOAD MÔ HÌNH VỚI 8-BIT QUANTIZATION
# ==========================================
print("Loading model in 8-bit...")
# Chuyển mô hình về 8-bit để tiết kiệm VRAM
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
)

local_rank = int(os.environ.get("LOCAL_RANK", 0))
device = f"cuda:{local_rank}"

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map={"": device}
    #device_map="auto" # Tự động phân bổ lên GPU
)

# Chuẩn bị mô hình cho việc huấn luyện quantized
model = prepare_model_for_kbit_training(model)

model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

# ==========================================
# 5. CẤU HÌNH LORA (PEFT)
# ==========================================
print("Applying LoRA...")
lora_config = LoraConfig(
    r=64,                     # Rank của ma trận (16 là mức cân bằng tốt cho 100k data)
    lora_alpha=128,            # Hệ số scale
    target_modules=["q_proj", "v_proj", "k_proj", "out_proj"], # Các module cần fine-tune trong Attention
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters() # Sẽ in ra số lượng tham số được train (chỉ khoảng 0.5% - 1%)

# ==========================================
# 6. HUẤN LUYỆN (TRAINING)
# ==========================================
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# training_args = Seq2SeqTrainingArguments(
#     output_dir=OUTPUT_DIR,
#     eval_strategy="steps",
#     eval_steps=1000,              # Đánh giá sau mỗi 1000 bước
#     save_strategy="steps",
#     save_steps=1000,
#     learning_rate=2e-4,           # Learning rate lớn hơn một chút cho LoRA (thường là 2e-4 hoặc 3e-4)
#     per_device_train_batch_size=8,# Giảm xuống 4 nếu báo lỗi OOM
#     per_device_eval_batch_size=8,
#     weight_decay=0.01,
#     save_total_limit=3,
#     num_train_epochs=3,           # 3 epochs cho 100k data là hợp lý để bắt đầu
#     predict_with_generate=False,  # Tắt sinh text lúc train để tiết kiệm VRAM & Thời gian
#     fp16=True,                    # Bật Mixed Precision
#     logging_steps=100,
#     optim="paged_adamw_8bit"      # Trình tối ưu hóa nhẹ cho GPU
# )

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="steps",
    eval_steps=5000,              
    save_strategy="steps",
    save_steps=5000,
    learning_rate=1e-4,           
    
    # --- ĐIỀU CHỈNH CHO MAX_LENGTH 256 & 2 GPU 12GB ---
    per_device_train_batch_size=2,  # Giữ ở 2, nếu vẫn OOM hãy hạ xuống 1
    per_device_eval_batch_size=2,   # QUAN TRỌNG: Phải hạ từ 8 xuống 2 để tránh treo khi Eval
    gradient_accumulation_steps=16, # Tăng lên 16 để Global Batch Size = 2 * 2 * 16 = 64
    # ------------------------------------------------
    
    warmup_steps=2000,              
    num_train_epochs=1,             
    weight_decay=0.01,
    save_total_limit=3,
    predict_with_generate=False,    # Vẫn nên để False khi train 1 triệu câu để tiết kiệm thời gian
    fp16=True,                    
    logging_steps=100,
    optim="paged_adamw_8bit",     
    ddp_find_unused_parameters=False,
    gradient_checkpointing=True,    # BẮT BUỘC để chạy được 256 tokens trên card 12GB
    
    # Tối ưu hóa thêm cho việc giải phóng bộ nhớ
    eval_accumulation_steps=1       # Giải phóng tensor đánh giá ngay lập tức để tiết kiệm VRAM
)


trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
)

print("Start training...")
trainer.train()

# Lưu PEFT Adapter (Rất nhẹ, chỉ khoảng chục MB)
trainer.save_model(f"{OUTPUT_DIR}/final_adapter_1m")
print("Training completed and saved!")