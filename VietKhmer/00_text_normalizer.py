import re
import unicodedata

class TextNormalizer:
    def __init__(self):
        # Từ điển chuẩn hóa dấu Tiếng Việt (Chuyển từ kiểu cũ sang kiểu mới)
        # Ví dụ: hòa -> hoà, thủy -> thuỷ
        self.vi_tone_mapping = {
            # ==========================================
            # 1. CHỮ THƯỜNG (Lowercase)
            # ==========================================
            # Nhóm 'oa'
            'òa': 'oà', 'óa': 'oá', 'ỏa': 'oả', 'õa': 'oã', 'ọa': 'oạ',
            # Nhóm 'oe'
            'òe': 'oè', 'óe': 'oé', 'ỏe': 'oẻ', 'õe': 'oẽ', 'ọe': 'oẹ',
            # Nhóm 'uy'
            'ùy': 'uỳ', 'úy': 'uý', 'ủy': 'uỷ', 'ũy': 'uỹ', 'ụy': 'uỵ',

            # ==========================================
            # 2. VIẾT HOA CHỮ ĐẦU (Capitalized)
            # ==========================================
            # Nhóm 'Oa'
            'Òa': 'Oà', 'Óa': 'Oá', 'Ỏa': 'Oả', 'Õa': 'Oã', 'Ọa': 'Oạ',
            # Nhóm 'Oe'
            'Òe': 'Oè', 'Óe': 'Oé', 'Ỏe': 'Oẻ', 'Õe': 'Oẽ', 'Ọe': 'Oẹ',
            # Nhóm 'Uy'
            'Ùy': 'Uỳ', 'Úy': 'Uý', 'Ủy': 'Uỷ', 'Ũy': 'Uỹ', 'Ụy': 'Uỵ',

            # ==========================================
            # 3. VIẾT HOA TOÀN BỘ (Uppercase)
            # ==========================================
            # Nhóm 'OA'
            'ÒA': 'OÀ', 'ÓA': 'OÁ', 'ỎA': 'OẢ', 'ÕA': 'OÃ', 'ỌA': 'OẠ',
            # Nhóm 'OE'
            'ÒE': 'OÈ', 'ÓE': 'OÉ', 'ỎE': 'OẺ', 'ÕE': 'OẼ', 'ỌE': 'OẸ',
            # Nhóm 'UY'
            'ÙY': 'UỲ', 'ÚY': 'UÝ', 'ỦY': 'UỶ', 'ŨY': 'UỸ', 'ỤY': 'UỴ',

            # ==========================================
            # 4. LỖI GÕ SHIFT LỆCH (Edge cases in web data)
            # Ví dụ người dùng gõ hÒA -> hOÀ
            # ==========================================
            'òA': 'oÀ', 'óA': 'oÁ', 'ỏA': 'oẢ', 'õA': 'oÃ', 'ọA': 'oẠ',
            'òE': 'oÈ', 'óE': 'oÉ', 'ỏE': 'oẺ', 'õE': 'oẼ', 'ọE': 'oẸ',
            'ùY': 'uỲ', 'úY': 'uÝ', 'ủY': 'uỶ', 'ũY': 'uỸ', 'ụY': 'uỴ'
        }
        
        # Build regex từ dictionary để thay thế tốc độ cao
        self.vi_tone_pattern = re.compile(
            "|".join(re.escape(key) for key in self.vi_tone_mapping.keys())
        )

    def normalize_unicode(self, text: str) -> str:
        """
        Ép chuỗi về chuẩn NFC. 
        NFC (Normalization Form Canonical Composition) gom các ký tự tổ hợp 
        (ví dụ: e + ^ + ' = ế) thành một mã Unicode duy nhất.
        """
        if not text:
            return ""
        return unicodedata.normalize('NFC', text)

    def clean_common(self, text: str) -> str:
        """Loại bỏ khoảng trắng thừa, HTML tags, và các ký tự dính (non-breaking spaces)."""
        # Xóa non-breaking space và các khoảng trắng đặc biệt
        text = text.replace('\xa0', ' ')
        text = text.replace('\t', ' ')
        
        # Xóa HTML tags nếu có rớt lại từ quá trình cào dữ liệu
        text = re.sub(r'<[^>]+>', '', text)
        
        # Gộp nhiều khoảng trắng thành 1 khoảng trắng duy nhất
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

    def process_vietnamese(self, text: str) -> str:
        """Pipeline xử lý riêng cho Tiếng Việt."""
        text = self.normalize_unicode(text)
        text = self.clean_common(text)
        
        # Chuẩn hóa vị trí dấu thanh
        text = self.vi_tone_pattern.sub(
            lambda match: self.vi_tone_mapping[match.group(0)], text
        )
        
        # Sửa lỗi dính dấu câu phổ biến (ví dụ: "chữ,chữ" -> "chữ, chữ")
        text = re.sub(r'([a-zA-Záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ])([,.?!:;])([a-zA-Záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ])', r'\1\2 \3', text)
        
        return text

    def process_khmer(self, text: str) -> str:
        """Pipeline xử lý riêng cho Tiếng Khmer."""
        text = self.normalize_unicode(text)
        text = self.clean_common(text)
        
        # Danh sách các ký tự vô hình cần loại bỏ
        invisible_chars = [
            '\u200b',  # Zero-Width Space (ZWSP) - Rất phổ biến trong Khmer
            '\u200c',  # Zero-Width Non-Joiner (ZWNJ)
            '\u200d',  # Zero-Width Joiner (ZWJ)
            '\ufeff',  # Byte Order Mark (BOM)
            '\u2028',  # Line separator
            '\u2029'   # Paragraph separator
        ]
        for char in invisible_chars:
            text = text.replace(char, '')
            
        # Sửa lỗi gõ dư nhiều ký tự Coeng (chân chữ - \u17D2) liên tiếp
        # \u17D2 dùng để tạo chân chữ, nếu gõ 2 lần liên tiếp sẽ gây lỗi hiển thị và nhiễu model
        text = re.sub(r'\u17D2{2,}', '\u17D2', text)
        
        # Sửa lỗi dấu chấm Khmer (Khan - ។) bị dính liền với chữ phía sau (nếu có)
        text = re.sub(r'(។)([^ \n])', r'\1 \2', text)

        return text

    def process_file(self, input_path: str, output_path: str, lang: str):
        """Xử lý hàng loạt đọc/ghi file an toàn."""
        print(f"🔄 Đang chuẩn hóa file {lang.upper()}: {input_path}")
        
        with open(input_path, 'r', encoding='utf-8') as fin, \
             open(output_path, 'w', encoding='utf-8') as fout:
            
            for line in fin:
                if not line.strip():
                    fout.write('\n')
                    continue
                
                if lang == 'vi':
                    cleaned_line = self.process_vietnamese(line)
                elif lang == 'km':
                    cleaned_line = self.process_khmer(line)
                else:
                    cleaned_line = self.clean_common(self.normalize_unicode(line))
                
                fout.write(cleaned_line + '\n')
                
        print(f"✅ Đã chuẩn hóa xong. File lưu tại: {output_path}")

# ==========================================
# CÁCH SỬ DỤNG SCRIPT
# ==========================================
if __name__ == "__main__":
    normalizer = TextNormalizer()
    
    # --- Test chuỗi đơn lẻ ---
    # Chú ý chuỗi Tiếng Việt: Chữ "hòa" dùng dấu cũ, dư khoảng trắng, dính dấu phẩy
    vi_raw = "Cộng   hòa xã hội chủ nghĩa Việt Nam,Độc lập - Tự do - Hạnh phúc."
    vi_clean = normalizer.process_vietnamese(vi_raw)
    print("VI Gốc  :", vi_raw)
    print("VI Sạch :", vi_clean) # Kết quả: "Cộng hoà xã hội chủ nghĩa Việt Nam, Độc lập - Tự do - Hạnh phúc."
    print("-" * 50)

    # Chú ý chuỗi Khmer: Chứa ZWSP ẩn giữa các chữ
    km_raw = "សួស្តី​! ទីក្រុង​ហូជីមិញ​ថ្ងៃនេះ​ស្អាត​ណាស់។តើអ្នកចង់ដើរលេងទេ?"
    km_clean = normalizer.process_khmer(km_raw)
    print("KM Gốc  :", repr(km_raw)) # Dùng repr để thấy các ký tự ẩn \u200b
    print("KM Sạch :", repr(km_clean))
    print("-" * 50)

    # --- Chạy xử lý trên File ---
    # Bỏ comment các dòng dưới đây và thay tên file để chạy thực tế
    # normalizer.process_file("raw_vietnamese.txt", "norm_vietnamese.txt", lang="vi")
    # normalizer.process_file("raw_khmer.txt", "norm_khmer.txt", lang="km")