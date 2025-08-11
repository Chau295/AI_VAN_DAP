# qna/management/commands/run_workers.py
import logging
import json
import os
import re
import asyncio
import subprocess
from unicodedata import normalize

import torch
from transformers import AutoTokenizer, AutoModel
from torch.nn.functional import cosine_similarity
import google.generativeai as genai
from faster_whisper import WhisperModel

from django.core.management.base import BaseCommand
from django.conf import settings
from channels.layers import get_channel_layer
from asgiref.sync import sync_to_async
from qna.models import Question

logger = logging.getLogger(__name__)


# --- CÁC HÀM XỬ LÝ NLP VÀ AI (CẬP NHẬT TỪ NOTEBOOK) ---

def preprocess_text_vietnamese(text):
    text = text.lower()
    text = normalize('NFC', text)
    text = re.sub(r'[^\w\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def get_sentence_embedding(text, tokenizer, model, device):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)


def rephrase_text_with_gemini(text, question_barem, gemini_model):
    """
    Sử dụng Gemini để sửa lỗi chính tả/ngữ pháp cho transcript từ Whisper.
    Logic được lấy từ file Untitled0.ipynb.
    """
    if not text or gemini_model is None:
        return text

    # Trích xuất các key points từ barem để làm ngữ cảnh
    barem_text = "\n".join([f"- {kp['text']}" for kp in question_barem['key_points']])
    prompt = f"""Văn bản sau là câu trả lời của sinh viên, được nhận dạng từ giọng nói. Dựa vào câu hỏi và barem điểm, hãy sửa lỗi chính tả, ngữ pháp và những từ bị nhận dạng sai do giọng địa phương. Chỉ sửa những lỗi thực sự cần thiết để câu trả lời khớp với barem, không thay đổi cấu trúc câu hay diễn đạt nếu nó không sai. Trả về duy nhất văn bản đã được sửa.

--- CÂU HỎI ---
{question_barem['question']}

--- BAREM ĐIỂM THAM KHẢO ---
{barem_text}

--- CÂU TRẢ LỜI CỦA SINH VIÊN (VĂN BẢN GỐC TỪ ASR) ---
{text}
"""
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Lỗi khi rephrase văn bản: {e}")
        return text


def score_traditional(student_answer, barem_key_points, tokenizer, model, device, threshold=0.6):
    """
    Chấm điểm bằng PhoBERT và trả về kết quả chi tiết cho từng ý.
    """
    student_embedding = get_sentence_embedding(preprocess_text_vietnamese(student_answer), tokenizer, model, device)
    total_score = 0.0
    detailed_results = []

    for kp in barem_key_points:
        kp_embedding = get_sentence_embedding(preprocess_text_vietnamese(kp["text"]), tokenizer, model, device)
        similarity = cosine_similarity(student_embedding, kp_embedding).item()

        is_matched = similarity >= threshold
        score_achieved = kp["weight"] if is_matched else 0.0
        total_score += score_achieved

        detailed_results.append({
            "text": kp["text"],
            "similarity": round(similarity, 2),
            "matched": is_matched
        })

    final_score = min(total_score, 10.0)
    return final_score, detailed_results


def score_with_gemini(student_answer, question_text, barem_key_points, gemini_model):
    """
    Chấm điểm bằng Gemini và trả về kết quả dưới dạng JSON.
    """
    prompt_parts = [
        "Bạn là một chuyên gia chấm điểm câu trả lời vấn đáp. Nhiệm vụ của bạn là chấm điểm câu trả lời của sinh viên dựa trên barem điểm đã cho một cách nghiêm ngặt.",
        "Đầu ra của bạn PHẢI là một chuỗi JSON hợp lệ có cấu trúc: {{\"diem_so\": float, \"phan_hoi\": \"string\", \"phan_tich\": [{{\"text\": \"string\", \"matched\": boolean}}]}}. Không thêm bất kỳ văn bản nào ngoài chuỗi JSON này.",
        f"--- CÂU HỎI ---",
        f"Câu hỏi: {question_text}",
        f"--- BAREM ĐIỂM ---"
    ]
    for kp in barem_key_points:
        prompt_parts.append(f"- {kp['text']} (Trọng số: {kp['weight']:.1f})")

    prompt_parts.extend([
        "--- CÂU TRẢ LỜI CỦA SINH VIÊN ---",
        student_answer,
        "--- HÃY CHẤM ĐIỂM VÀ ĐƯA RA KẾT QUẢ DƯỚI DẠNG JSON ---"
    ])
    full_prompt = "\n".join(prompt_parts)

    try:
        response = gemini_model.generate_content(full_prompt)
        match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            return {"score": float(data.get("diem_so", 0.0)), "feedback": data.get("phan_hoi", "Không có phản hồi."),
                    "analysis": data.get("phan_tich", [])}
    except Exception as e:
        logger.error(f"Lỗi Gemini: {e}")
    return {"score": 0.0, "feedback": "Lỗi hệ thống chấm điểm AI.", "analysis": []}


def convert_webm_to_wav(webm_path):
    """
    Chuyển đổi file âm thanh từ định dạng webm sang wav.
    """
    wav_path = webm_path.replace(".webm", ".wav")
    try:
        # Lệnh ffmpeg để chuyển đổi, -y để tự động ghi đè file đã có
        command = ["ffmpeg", "-i", webm_path, "-ac", "1", "-ar", "16000", "-y", wav_path]
        subprocess.run(command, check=True, capture_output=True, text=True)
        logger.info(f"Đã chuyển đổi thành công {webm_path} sang {wav_path}")
        return wav_path
    except subprocess.CalledProcessError as e:
        logger.error(f"Lỗi khi chạy ffmpeg: {e.stderr}")
        return None
    except FileNotFoundError:
        logger.error("Lỗi: Lệnh 'ffmpeg' không được tìm thấy. Hãy đảm bảo ffmpeg đã được cài đặt và thêm vào PATH.")
        return None


class Command(BaseCommand):
    help = 'Chạy worker lắng nghe các tác vụ AI từ channel layer'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.channel_layer = get_channel_layer()
        self.audio_chunks = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if torch.cuda.is_available() else "int8"

        self.stdout.write(f"Sử dụng thiết bị: {self.device} với kiểu tính toán {self.compute_type}")

        self.stdout.write("Đang tải Whisper model (sử dụng faster-whisper)...")
        self.whisper_model = WhisperModel("large", device=self.device, compute_type=self.compute_type)
        self.stdout.write("✅ Whisper đã sẵn sàng.")

        self.stdout.write("Đang tải PhoBERT model...")
        self.phobert_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
        self.phobert_model = AutoModel.from_pretrained("vinai/phobert-base").to(self.device)
        self.phobert_model.eval()
        self.stdout.write("✅ PhoBERT đã sẵn sàng.")

        self.stdout.write("Đang cấu hình Gemini model...")
        try:
            api_key = os.environ.get('GOOGLE_API_KEY')
            if not api_key:
                raise ValueError("Biến môi trường GOOGLE_API_KEY chưa được thiết lập.")
            genai.configure(api_key=api_key)
            self.gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')
            self.stdout.write("✅ Gemini đã sẵn sàng.")
        except Exception as e:
            self.gemini_model = None
            self.stderr.write(self.style.ERROR(f"Lỗi cấu hình Gemini: {e}"))

    @sync_to_async
    def get_question_and_barem(self, question_id):
        try:
            question_obj = Question.objects.select_related('subject').get(pk=question_id)
            quiz_file_path = os.path.join(settings.BASE_DIR, 'quiz_data', question_obj.subject.quiz_data_file)
            with open(quiz_file_path, 'r', encoding='utf-8') as f:
                all_barems = json.load(f)
            for barem in all_barems:
                if barem['id'] == question_obj.question_id_in_barem:
                    return barem
        except Exception as e:
            logger.error(f"Không thể tải barem cho câu hỏi {question_id}: {e}")
        return None

    async def process_end(self, message):
        reply_channel = message['reply_channel']
        question_id = message['question_id']
        chunks = self.audio_chunks.pop(reply_channel, [])
        if not chunks:
            logger.warning("Không có chunk âm thanh nào để xử lý.")
            return

        safe_channel_name = re.sub(r'[^a-zA-Z0-9]', '_', reply_channel)
        webm_path = os.path.join(settings.BASE_DIR, f'temp_audio_{safe_channel_name}.webm')

        logger.info(f"Đang ghi {len(chunks)} chunk vào file {webm_path}...")
        with open(webm_path, 'wb') as f:
            for chunk in chunks:
                f.write(chunk)

        wav_path = await asyncio.to_thread(convert_webm_to_wav, webm_path)
        if os.path.exists(webm_path):
            os.remove(webm_path)

        if not wav_path:
            await self.channel_layer.send(reply_channel, {'type': 'exam.result', 'message': {
                'error': 'Lỗi xử lý file âm thanh. Vui lòng thử lại.'}})
            return

        logger.info(f"Đang nhận dạng giọng nói từ file: {wav_path}...")
        try:
            segments, _ = await asyncio.to_thread(self.whisper_model.transcribe, wav_path, language="vi")
            transcript_parts = [segment.text for segment in segments]
            raw_transcript = "".join(transcript_parts).strip()
        except Exception as e:
            logger.error(f"Lỗi khi nhận dạng giọng nói: {e}")
            raw_transcript = ""
        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)

        if not raw_transcript:
            await self.channel_layer.send(reply_channel, {'type': 'exam.result', 'message': {
                'error': 'Không nhận dạng được giọng nói. Vui lòng thử lại.'}})
            return

        logger.info(f"Transcript gốc: '{raw_transcript}'")

        question_barem = await self.get_question_and_barem(question_id)
        if not question_barem:
            await self.channel_layer.send(reply_channel, {'type': 'exam.result', 'message': {
                'error': 'Lỗi hệ thống: Không tìm thấy barem chấm điểm.'}})
            return

        logger.info("Đang dùng Gemini để sửa lỗi transcript...")
        rephrased_transcript = await asyncio.to_thread(rephrase_text_with_gemini, raw_transcript, question_barem,
                                                       self.gemini_model)
        logger.info(f"Transcript đã sửa: '{rephrased_transcript}'")

        traditional_task = asyncio.to_thread(score_traditional, rephrased_transcript, question_barem['key_points'],
                                             self.phobert_tokenizer, self.phobert_model, self.device)
        gemini_task = asyncio.to_thread(score_with_gemini, rephrased_transcript, question_barem['question'],
                                        question_barem['key_points'], self.gemini_model)

        (traditional_score, traditional_analysis), gemini_result = await asyncio.gather(traditional_task, gemini_task)

        pho_bert_analysis_map = {item['text']: item for item in traditional_analysis}
        combined_analysis = []
        for gemini_item in gemini_result.get('analysis', []):
            pho_bert_item = pho_bert_analysis_map.get(gemini_item['text'])
            if pho_bert_item:
                gemini_item['phoBERT_matched'] = pho_bert_item['matched']
                gemini_item['phoBERT_similarity'] = pho_bert_item['similarity']
            combined_analysis.append(gemini_item)

        final_score = (gemini_result['score'] * 0.7) + (traditional_score * 0.3)

        final_result = {"question_id": question_id, "transcript": rephrased_transcript,
                        "final_score": round(final_score, 2), "feedback": gemini_result.get('feedback', ''),
                        "analysis": combined_analysis}

        await self.channel_layer.send(reply_channel, {'type': 'exam.result', 'message': final_result})

    async def process_chunk(self, message):
        reply_channel = message['reply_channel']
        if reply_channel not in self.audio_chunks:
            self.audio_chunks[reply_channel] = []
        self.audio_chunks[reply_channel].append(message['audio_chunk'])

    async def run(self):
        logger.info("Worker đang lắng nghe trên kênh 'asr-tasks'...")
        while True:
            message = await self.channel_layer.receive('asr-tasks')
            task_type = message.get('type')
            if task_type == 'asr.process.chunk':
                await self.process_chunk(message)
            elif task_type == 'asr.process.end':
                asyncio.create_task(self.process_end(message))

    def handle(self, *args, **options):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.stdout.write(self.style.SUCCESS('Starting AI Worker...'))
        try:
            asyncio.run(self.run())
        except KeyboardInterrupt:
            self.stdout.write(self.style.WARNING('Worker stopped by user.'))