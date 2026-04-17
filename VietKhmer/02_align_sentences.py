import numpy as np
import jsonlines
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 1. CẤU HÌNH
# ==========================================
# Tải mô hình LaBSE (Hỗ trợ 109 ngôn ngữ bao gồm Tiếng Việt và Khmer)
print("⏳ Đang tải mô hình LaBSE...")
model = SentenceTransformer('sentence-transformers/LaBSE')

# Ngưỡng tương đồng (Threshold): Từ 0.0 đến 1.0
# Đặt 0.6 hoặc 0.65 để đảm bảo cặp câu có nghĩa thực sự khớp nhau
SIMILARITY_THRESHOLD = 0.6 

FILE_VI = "/mnt/e/AILAB/TOOLKITS/TrainTextTranslate/VietKhmer/raw_vietnamese.txt"
FILE_KM = "/mnt/e/AILAB/TOOLKITS/TrainTextTranslate/VietKhmer/raw_khmer.txt"
OUTPUT_FILE = "/mnt/e/AILAB/TOOLKITS/TrainTextTranslate/VietKhmer/aligned_vi_km.jsonl"

# ==========================================
# 2. ĐỌC VÀ CHIA CÂU (Tự động loại bỏ dòng trống)
# ==========================================
def load_sentences(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        # Tách theo dòng, loại bỏ khoảng trắng thừa
        sentences = [line.strip() for line in f if line.strip()]
    return sentences

print("📂 Đang đọc dữ liệu thô...")
vi_sentences = load_sentences(FILE_VI)
km_sentences = load_sentences(FILE_KM)

print(f"Số câu Tiếng Việt: {len(vi_sentences)}")
print(f"Số câu Tiếng Khmer: {len(km_sentences)}")

# ==========================================
# 3. TRÍCH XUẤT VECTOR (EMBEDDINGS)
# ==========================================
print("🧠 Đang chuyển đổi câu thành Vectors...")
vi_embeddings = model.encode(vi_sentences, show_progress_bar=True)
km_embeddings = model.encode(km_sentences, show_progress_bar=True)

# Tính ma trận độ tương đồng Cosine giữa mọi câu Vi và câu Km
print("🧮 Đang tính toán ma trận tương đồng...")
sim_matrix = cosine_similarity(vi_embeddings, km_embeddings)

# ==========================================
# 4. THUẬT TOÁN QUY HOẠCH ĐỘNG (ALIGNMENT DP)
# Thuật toán này giữ nguyên thứ tự câu của tài liệu (giống Hunalign)
# ==========================================
print("🔗 Đang khớp câu (Alignment)...")
n = len(vi_sentences)
m = len(km_sentences)

# Bảng DP lưu điểm số tối đa
dp = np.zeros((n + 1, m + 1))
# Bảng lưu vết để truy hồi
path = np.zeros((n + 1, m + 1, 2), dtype=int)

for i in range(1, n + 1):
    for j in range(1, m + 1):
        # Trường hợp 1: Khớp câu i với câu j
        score_match = dp[i-1][j-1] + sim_matrix[i-1][j-1]
        
        # Trường hợp 2: Bỏ qua câu tiếng Việt thứ i
        score_skip_vi = dp[i-1][j]
        
        # Trường hợp 3: Bỏ qua câu tiếng Khmer thứ j
        score_skip_km = dp[i][j-1]
        
        # Chọn chiến lược tốt nhất
        best = max(score_match, score_skip_vi, score_skip_km)
        dp[i][j] = best
        
        if best == score_match:
            path[i][j] = [i-1, j-1]
        elif best == score_skip_vi:
            path[i][j] = [i-1, j]
        else:
            path[i][j] = [i, j-1]

# Truy hồi (Backtracking) để lấy kết quả
aligned_pairs = []
i, j = n, m
while i > 0 and j > 0:
    prev_i, prev_j = path[i][j]
    if prev_i == i - 1 and prev_j == j - 1:
        sim_score = sim_matrix[i-1][j-1]
        # Chỉ giữ lại các cặp vượt qua ngưỡng độ tin cậy
        if sim_score >= SIMILARITY_THRESHOLD:
            aligned_pairs.append({
                "vi": vi_sentences[i-1],
                "km": km_sentences[j-1],
                "score": round(float(sim_score), 4)
            })
    i, j = prev_i, prev_j

# Đảo ngược danh sách vì đang truy hồi từ cuối lên
aligned_pairs = aligned_pairs[::-1]

# ==========================================
# 5. LƯU KẾT QUẢ ĐỂ FINE-TUNE
# ==========================================
print(f"✅ Đã tìm thấy {len(aligned_pairs)} cặp câu khớp nhau thành công!")

with jsonlines.open(OUTPUT_FILE, mode='w') as writer:
    for pair in aligned_pairs:
        writer.write(pair)

print(f"📁 Dữ liệu đã được lưu chuẩn xác vào: {OUTPUT_FILE}")
print("Mẫu 1 cặp câu đầu tiên:", aligned_pairs[0] if aligned_pairs else "Không có")