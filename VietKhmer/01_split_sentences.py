import re
from underthesea import sent_tokenize

# ==========================================
# 1. HÀM TÁCH CÂU TIẾNG VIỆT
# ==========================================
def process_vietnamese_text(input_filepath, output_filepath):
    print(f"🔄 Đang xử lý file Tiếng Việt: {input_filepath}")
    
    with open(input_filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Dùng thư viện underthesea để tách câu (xử lý tốt các từ viết tắt như TP.HCM, GS.TS)
    sentences = sent_tokenize(content)
    
    # Lọc bỏ các dòng trống và khoảng trắng thừa
    clean_sentences = [s.strip() for s in sentences if s.strip()]

    with open(output_filepath, 'w', encoding='utf-8') as f:
        for sentence in clean_sentences:
            f.write(sentence + '\n')
            
    print(f"✅ Đã tách được {len(clean_sentences)} câu. Lưu tại: {output_filepath}")

# ==========================================
# 2. HÀM TÁCH CÂU TIẾNG KHMER
# ==========================================
def process_khmer_text(input_filepath, output_filepath):
    print(f"🔄 Đang xử lý file Tiếng Khmer: {input_filepath}")
    
    with open(input_filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Xóa ký tự Zero-Width Space (\u200b) thường bị ẩn trong văn bản Khmer
    content = content.replace('\u200b', '')

    # Regex tách câu: Tách ngay sau ký tự Khan (។), chấm hỏi (?), chấm than (!), hoặc xuống dòng
    # (?<=[។?!]) có nghĩa là lookbehind, giữ lại dấu câu ở cuối câu.
    raw_sentences = re.split(r'(?<=[។?!])\s*|\n+', content)
    
    # Lọc bỏ các dòng trống và khoảng trắng thừa
    clean_sentences = [s.strip() for s in raw_sentences if s.strip()]

    with open(output_filepath, 'w', encoding='utf-8') as f:
        for sentence in clean_sentences:
            f.write(sentence + '\n')

    print(f"✅ Đã tách được {len(clean_sentences)} câu. Lưu tại: {output_filepath}")

# ==========================================
# 3. CHẠY THỰC THI (THAY ĐỔI TÊN FILE CỦA BẠN Ở ĐÂY)
# ==========================================
if __name__ == "__main__":
    # File đầu vào (chứa các đoạn văn dài)
    vietnamese_input = "/mnt/e/AILAB/TOOLKITS/TrainTextTranslate/VietKhmer/draft_vietnamese.txt"
    khmer_input = "/mnt/e/AILAB/TOOLKITS/TrainTextTranslate/VietKhmer/draft_khmer.txt"

    # File đầu ra (để đưa vào tool LaBSE Alignment ở bước trước)
    vietnamese_output = "/mnt/e/AILAB/TOOLKITS/TrainTextTranslate/VietKhmer/raw_vietnamese.txt"
    khmer_output = "/mnt/e/AILAB/TOOLKITS/TrainTextTranslate/VietKhmer/raw_khmer.txt"

    # Tạo file nháp giả lập để test nếu bạn chưa có file
    # with open(vietnamese_input, 'w', encoding='utf-8') as f:
    #     f.write("Xin chào! TP.HCM hôm nay rất đẹp. Bạn có muốn đi dạo không?\nTôi rất thích thời tiết này.")
    # with open(khmer_input, 'w', encoding='utf-8') as f:
    #     f.write("សួស្តី! ទីក្រុងហូជីមិញថ្ងៃនេះស្អាតណាស់។ តើអ្នកចង់ដើរលេងទេ?\nខ្ញុំចូលចិត្តអាកាសធាតុនេះណាស់។")

    print("-" * 40)
    process_vietnamese_text(vietnamese_input, vietnamese_output)
    print("-" * 40)
    process_khmer_text(khmer_input, khmer_output)
    print("-" * 40)
    print("Hoàn tất! Bạn có thể dùng 2 file output này để chạy script alignment.")