#pip install fastapi uvicorn python-multipart pydantic underthesea sentence-transformers scikit-learn numpy jsonlines

import re
import os
import unicodedata
import numpy as np
import jsonlines
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager

from underthesea import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 1. MODULE CHUẨN HÓA VĂN BẢN
# ==========================================
class TextNormalizer:
    def __init__(self):
        self.vi_tone_mapping = {
            'òa': 'oà', 'óa': 'oá', 'ỏa': 'oả', 'õa': 'oã', 'ọa': 'oạ',
            'òe': 'oè', 'óe': 'oé', 'ỏe': 'oẻ', 'õe': 'oẽ', 'ọe': 'oẹ',
            'ùy': 'uỳ', 'úy': 'uý', 'ủy': 'uỷ', 'ũy': 'uỹ', 'ụy': 'uỵ',
            'Òa': 'Oà', 'Óa': 'Oá', 'Ỏa': 'Oả', 'Õa': 'Oã', 'Ọa': 'Oạ',
            'Òe': 'Oè', 'Óe': 'Oé', 'Ỏe': 'Oẻ', 'Õe': 'Oẽ', 'Ọe': 'Oẹ',
            'Ùy': 'Uỳ', 'Úy': 'Uý', 'Ủy': 'Uỷ', 'Ũy': 'Uỹ', 'Ụy': 'Uỵ',
            'ÒA': 'OÀ', 'ÓA': 'OÁ', 'ỎA': 'OẢ', 'ÕA': 'OÃ', 'ỌA': 'OẠ',
            'ÒE': 'OÈ', 'ÓE': 'OÉ', 'ỎE': 'OẺ', 'ÕE': 'OẼ', 'ỌE': 'OẸ',
            'ÙY': 'UỲ', 'ÚY': 'UÝ', 'ỦY': 'UỶ', 'ŨY': 'UỸ', 'ỤY': 'UỴ',
            'òA': 'oÀ', 'óA': 'oÁ', 'ỏA': 'oẢ', 'õA': 'oÃ', 'ọA': 'oẠ',
            'òE': 'oÈ', 'óE': 'oÉ', 'ỏE': 'oẺ', 'õE': 'oẼ', 'ọE': 'oẸ',
            'ùY': 'uỲ', 'úY': 'uÝ', 'ủY': 'uỶ', 'ũY': 'uỸ', 'ụY': 'uỴ'
        }
        self.vi_tone_pattern = re.compile("|".join(re.escape(key) for key in self.vi_tone_mapping.keys()))

    def clean_common(self, text: str) -> str:
        if not text: return ""
        text = unicodedata.normalize('NFC', text)
        text = text.replace('\xa0', ' ').replace('\t', ' ')
        text = re.sub(r'<[^>]+>', '', text)
        return re.sub(r'\s+', ' ', text).strip()

    def process_vietnamese(self, text: str) -> str:
        text = self.clean_common(text)
        text = self.vi_tone_pattern.sub(lambda match: self.vi_tone_mapping[match.group(0)], text)
        text = re.sub(r'([a-zA-Záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ])([,.?!:;])([a-zA-Záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ])', r'\1\2 \3', text)
        return text

    def process_khmer(self, text: str) -> str:
        text = self.clean_common(text)
        for char in ['\u200b', '\u200c', '\u200d', '\ufeff', '\u2028', '\u2029']:
            text = text.replace(char, '')
        text = re.sub(r'\u17D2{2,}', '\u17D2', text)
        text = re.sub(r'(។)([^ \n])', r'\1 \2', text)
        return text

# ==========================================
# 2. MODULE TÁCH CÂU (TOKENIZER)
# ==========================================
class SentenceTokenizer:
    @staticmethod
    def process_vietnamese(text: str) -> list:
        sentences = sent_tokenize(text)
        return [s.strip() for s in sentences if s.strip()]

    @staticmethod
    def process_khmer(text: str) -> list:
        raw_sentences = re.split(r'(?<=[។?!])\s*|\n+', text)
        return [s.strip() for s in raw_sentences if s.strip()]

# ==========================================
# 3. MODULE GIÓNG HÀNG CÂU (ALIGNMENT)
# ==========================================
class SentenceAligner:
    def __init__(self):
        self.model = None

    def load_model(self):
        print("⏳ Đang tải mô hình LaBSE (Khoảng 1.8GB)...")
        self.model = SentenceTransformer('sentence-transformers/LaBSE')
        print("✅ Đã tải xong mô hình LaBSE!")

    def align(self, source_sentences: list, target_sentences: list, threshold: float = 0.6) -> list:
        if not source_sentences or not target_sentences: return []

        source_embeddings = self.model.encode(source_sentences, show_progress_bar=False)
        target_embeddings = self.model.encode(target_sentences, show_progress_bar=False)

        sim_matrix = cosine_similarity(source_embeddings, target_embeddings)
        n, m = len(source_sentences), len(target_sentences)

        dp = np.zeros((n + 1, m + 1))
        path = np.zeros((n + 1, m + 1, 2), dtype=int)

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                score_match = dp[i-1][j-1] + sim_matrix[i-1][j-1]
                score_skip_vi = dp[i-1][j]
                score_skip_km = dp[i][j-1]

                best = max(score_match, score_skip_vi, score_skip_km)
                dp[i][j] = best

                if best == score_match: path[i][j] = [i-1, j-1]
                elif best == score_skip_vi: path[i][j] = [i-1, j]
                else: path[i][j] = [i, j-1]

        aligned_pairs = []
        i, j = n, m
        while i > 0 and j > 0:
            prev_i, prev_j = path[i][j]
            if prev_i == i - 1 and prev_j == j - 1:
                sim_score = sim_matrix[i-1][j-1]
                if sim_score >= threshold:
                    aligned_pairs.append({
                        "vi": source_sentences[i-1],
                        "km": target_sentences[j-1],
                        "score": round(float(sim_score), 4)
                    })
            i, j = prev_i, prev_j

        return aligned_pairs[::-1]

# ==========================================
# 4. KHỞI TẠO FASTAPI & PIPELINE
# ==========================================
normalizer = TextNormalizer()
tokenizer = SentenceTokenizer()
aligner = SentenceAligner()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model LaBSE 1 lần duy nhất khi khởi động Server
    aligner.load_model()
    yield

app = FastAPI(
    title="Viet-Khmer Data Processing Pipeline",
    description="End-to-end API: Chuẩn hóa -> Tách câu -> Alignment",
    lifespan=lifespan
)

class PipelineRequest(BaseModel):
    vi_text: str
    km_text: str
    threshold: float = 0.6

class PipelineResponse(BaseModel):
    total_pairs: int
    aligned_data: list[dict]

# ==========================================
# 5. CÁC ENDPOINT API
# ==========================================

@app.post("/api/pipeline/text", response_model=PipelineResponse, tags=["Full Pipeline"])
async def process_text_pipeline(req: PipelineRequest):
    """Xử lý trực tiếp 2 đoạn text thô (Viet và Khmer) thành các cặp câu song ngữ."""
    # Bước 1: Chuẩn hóa
    norm_vi = normalizer.process_vietnamese(req.vi_text)
    norm_km = normalizer.process_khmer(req.km_text)

    # Bước 2: Tách câu
    sentences_vi = tokenizer.process_vietnamese(norm_vi)
    sentences_km = tokenizer.process_khmer(norm_km)

    # Bước 3: Alignment
    aligned_pairs = aligner.align(sentences_vi, sentences_km, req.threshold)
    
    return PipelineResponse(total_pairs=len(aligned_pairs), aligned_data=aligned_pairs)

@app.post("/api/pipeline/file", tags=["Full Pipeline"])
async def process_file_pipeline(
    vi_file: UploadFile = File(..., description="File text Tiếng Việt thô"),
    km_file: UploadFile = File(..., description="File text Tiếng Khmer thô"),
    threshold: float = Form(0.6, description="Ngưỡng tương đồng Cosine (0.0 -> 1.0)")
):
    """Upload 2 file thô, tự động xử lý toàn bộ quy trình và trả về file aligned.jsonl"""
    try:
        vi_raw = (await vi_file.read()).decode('utf-8')
        km_raw = (await km_file.read()).decode('utf-8')

        # Chạy qua Pipeline
        norm_vi = normalizer.process_vietnamese(vi_raw)
        norm_km = normalizer.process_khmer(km_raw)

        sentences_vi = tokenizer.process_vietnamese(norm_vi)
        sentences_km = tokenizer.process_khmer(norm_km)

        aligned_pairs = aligner.align(sentences_vi, sentences_km, threshold)

        # Lưu kết quả
        os.makedirs("temp", exist_ok=True)
        output_filename = "aligned_vi_km.jsonl"
        output_path = os.path.join("temp", output_filename)

        with jsonlines.open(output_path, mode='w') as writer:
            for pair in aligned_pairs:
                writer.write(pair)

        return FileResponse(path=output_path, filename=output_filename, media_type='application/jsonl')

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Không để reload=True vì sẽ load lại model LaBSE mỗi lần đổi code rất lâu
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)