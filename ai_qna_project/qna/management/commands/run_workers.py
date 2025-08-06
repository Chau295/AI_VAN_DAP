# qna/management/commands/run_workers.py
import logging
import json
import os
import re
import asyncio
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
    student_embedding = get_sentence_embedding(preprocess_text_vietnamese(student_answer), tokenizer, model, device)
    total_score = 0.0
    for kp in barem_key_points:
        kp_embedding = get_sentence_embedding(preprocess_text_vietnamese(kp["text"]), tokenizer, model, device)
        similarity = cosine_similarity(student_embedding, kp_embedding).item()
        if similarity >= threshold:
            total_score += kp["weight"]
    return min(total_score, 10.0)


def score_with_gemini(student_answer, question_text, barem_key_points, gemini_model):
    prompt_parts = [
        "Bạn là một chuyên gia chấm điểm câu trả lời vấn đáp. Nhiệm vụ của bạn là chấm điểm câu trả lời của sinh viên dựa trên barem điểm đã cho một cách nghiêm ngặt.",
        "Đầu ra của bạn PHẢI là một chuỗi JSON hợp lệ có cấu trúc: {\"diem_so\": float, \"phan_hoi\": \"string\", \"phan_tich\": [{\"text\": \"string\", \"matched\": boolean}]}. Không thêm bất kỳ văn bản nào ngoài chuỗi JSON này.",
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


class Command(BaseCommand):
    help = 'Chạy worker lắng nghe các tác vụ AI từ channel layer'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.channel_layer = get_channel_layer()
        self.audio_files = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if torch.cuda.is_available() else "int8"

        self.stdout.write(f"Sử dụng thiết bị: {self.device} với kiểu tính toán {self.compute_type}")

        self.stdout.write("Đang tải Whisper model (sử dụng faster-whisper)...")
        # === NÂNG CẤP MÔ HÌNH TẠI ĐÂY THEO YÊU CẦU CỦA BẠN ===
        self.whisper_model = WhisperModel("medium", device=self.device, compute_type=self.compute_type)
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
            self.gemini_model = genai.GenerativeModel('gemini-1.5-pro-latest')
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
                    return barem  # Trả về toàn bộ barem object
        except Exception as e:
            logger.error(f"Không thể tải barem cho câu hỏi {question_id}: {e}")
        return None

    async def process_end(self, message):
        reply_channel = message['reply_channel']
        question_id = message['question_id']
        audio_path = self.audio_files.pop(reply_channel, None)

        if not audio_path or not os.path.exists(audio_path):
            return

        try:
            file_size = os.path.getsize(audio_path)
            if file_size < 1024:
                logger.warning("File âm thanh quá nhỏ, có thể không có nội dung.")
                await self.channel_layer.send(reply_channel, {'type': 'exam.result', 'message': {
                    'error': 'Ghi âm quá ngắn hoặc không có âm thanh. Vui lòng thử lại.'}})
                if os.path.exists(audio_path): os.remove(audio_path)
                return
        except OSError as e:
            logger.error(f"Không thể kiểm tra kích thước file: {e}")
            return

        logger.info(f"Đang nhận dạng giọng nói từ file: {audio_path}...")
        try:
            segments, _ = await asyncio.to_thread(self.whisper_model.transcribe, audio_path, language="vi")
            transcript_parts = [segment.text for segment in segments]
            raw_transcript = "".join(transcript_parts).strip()
        except Exception as e:
            logger.error(f"Lỗi khi nhận dạng giọng nói: {e}")
            raw_transcript = ""
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)

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

        traditional_score, gemini_result = await asyncio.gather(traditional_task, gemini_task)

        final_score = (gemini_result['score'] * 0.7) + (traditional_score * 0.3)

        final_result = {"question_id": question_id, "transcript": rephrased_transcript,
                        "final_score": round(final_score, 2), "feedback": gemini_result.get('feedback', ''),
                        "analysis": gemini_result.get('analysis', [])}

        await self.channel_layer.send(reply_channel, {'type': 'exam.result', 'message': final_result})

    async def process_chunk(self, message):
        reply_channel = message['reply_channel']
        if reply_channel not in self.audio_files:
            safe_channel_name = re.sub(r'[^a-zA-Z0-9]', '_', reply_channel)
            temp_file_path = os.path.join(settings.BASE_DIR, f'temp_audio_{safe_channel_name}.webm')
            self.audio_files[reply_channel] = temp_file_path
            with open(temp_file_path, 'wb') as f:
                f.write(message['audio_chunk'])
        else:
            with open(self.audio_files[reply_channel], 'ab') as f:
                f.write(message['audio_chunk'])

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