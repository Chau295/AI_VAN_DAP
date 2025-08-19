# qna/management/commands/run_workers.py
import logging
import json
import os
import re
import asyncio
import subprocess


# --- Scoring helper for supplementary question (scale 0..10 to 0..1) ---
SUPP_MAX_PER_QUESTION = 1.0
SUPP_MAX_COUNT = 2

def _scale_supp(score_out_of_10: float) -> float:
    try:
        s = float(score_out_of_10 or 0.0)
    except (TypeError, ValueError):
        s = 0.0
    s = (s / 10.0) * SUPP_MAX_PER_QUESTION
    return max(0.0, min(s, SUPP_MAX_PER_QUESTION))
from unicodedata import normalize
from uuid import uuid4
import wave
from array import array
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModel
from torch.nn.functional import cosine_similarity
import openai

from django.core.management.base import BaseCommand
from django.conf import settings
from channels.layers import get_channel_layer
from asgiref.sync import sync_to_async
from django.core.exceptions import ObjectDoesNotExist
from qna.models import Question, ExamSession, ExamResult, SupplementaryResult

logger = logging.getLogger(__name__)

# ====== HẰNG SỐ / HELPERS ======
EBML_MAGIC = b"\x1A\x45\xDF\xA3"

def has_ebml_header(first_bytes: bytes) -> bool:
    return first_bytes.startswith(EBML_MAGIC)

def wav_duration_and_rms(path: str):
    """
    Đo thời lượng (giây) và RMS biên độ của WAV 16-bit mono 16k.
    """
    try:
        with wave.open(path, 'rb') as w:
            fr = w.getframerate()
            n = w.getnframes()
            sw = w.getsampwidth()
            ch = w.getnchannels()
            if fr <= 0 or n <= 0:
                return 0.0, 0.0
            duration = n / float(fr)
            raw = w.readframes(n)
        if sw != 2 or ch != 1:
            # không đúng định dạng mong muốn
            return duration, 0.0
        samples = array('h', raw)
        if not samples:
            return duration, 0.0
        acc = sum(float(s) * float(s) for s in samples)
        rms = (acc / len(samples)) ** 0.5
        return duration, rms
    except Exception as e:
        logger.error(f"Không thể phân tích WAV {path}: {e}")
        return 0.0, 0.0

def preprocess_text_vietnamese(text: str) -> str:
    text = text.lower()
    text = normalize('NFC', text)
    text = re.sub(r'[^\w\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def get_sentence_embedding(text, tokenizer, model, device):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    # (1, hidden)
    return outputs.last_hidden_state.mean(dim=1)

async def rephrase_text_with_chatgpt(text, question_text, client):
    if not text or client is None:
        return text
    system_prompt = "Bạn là một trợ lý ngôn ngữ chuyên sửa lỗi chính tả tiếng Việt một cách chính xác."
    user_prompt = f"""Văn bản sau là câu trả lời của sinh viên, được nhận dạng từ giọng nói. Nhiệm vụ của bạn là chỉ sửa các lỗi chính tả (ví dụ: 'chủng' thành 'chuẩn') do giọng nói vùng miền hoặc phát âm sai.
Tuyệt đối KHÔNG được loại bỏ các từ bị lặp, không thêm từ, và không thay đổi cấu trúc hay trật tự câu gốc.
Trả về duy nhất văn bản đã được sửa.

--- CÂU HỎI ---
{question_text}

--- CÂU TRẢ LỜI GỐC ---
{text}
"""
    try:
        resp = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Lỗi rephrase OpenAI: {e}")
        return text

def score_student_answer_traditional(student_answer_raw, question_barem, tokenizer, model, device, similarity_threshold=0.5):
    """
    Chấm điểm theo barem bằng PhoBERT (trọng số kết hợp).
    FIX: dùng cosine_similarity(dim=1) trên tensor shape (1, hidden) để tránh lỗi 768 elements.
    """
    if tokenizer is None or model is None:
        return 0.0
    student_answer_pre = preprocess_text_vietnamese(student_answer_raw)
    if not student_answer_pre:
        return 0.0

    student_emb = get_sentence_embedding(student_answer_pre, tokenizer, model, device)  # (1, h)
    total = 0.0
    for kp in question_barem.get("key_points", []):
        kp_pre = preprocess_text_vietnamese(kp.get("text", ""))
        kp_emb = get_sentence_embedding(kp_pre, tokenizer, model, device)  # (1, h)
        sim = cosine_similarity(student_emb, kp_emb, dim=1).item()  # scalar
        if sim >= similarity_threshold:
            total += float(kp.get("weight", 0.0))
    return min(total, float(question_barem.get("max_score", 10.0)))

async def score_student_answer_with_openai(student_answer_raw, question_barem, openai_client, model_name="gpt-4o-mini"):
    if openai_client is None:
        return 0.0, "OpenAI client chưa được khởi tạo."

    max_score = float(question_barem.get("max_score", 10.0))
    system_prompt = (
        "Bạn là một chuyên gia chấm điểm câu trả lời vấn đáp về khoa học dữ liệu. "
        "Chấm điểm nghiêm ngặt theo barem dưới đây. "
        f"Điểm tối đa cho câu hỏi này là {max_score:.2f}.\n"
        "ĐẦU RA BẮT BUỘC: một chuỗi JSON hợp lệ duy nhất có dạng "
        '{"diem_so": float, "phan_hoi": "string"}. '
        "Không thêm bất kỳ văn bản nào ngoài JSON."
    )
    barem_lines = [f"- {kp.get('id','KP')}: (Trọng số {kp.get('weight',0)}). {kp.get('text','')}" for kp in question_barem.get("key_points", [])]
    barem_block = "\n".join(barem_lines) if barem_lines else "(Không có key_points)"
    user_prompt = (
        f"--- CÂU HỎI ---\n{question_barem.get('question','')}\n\n"
        f"--- BAREM ---\n{barem_block}\n\n"
        f"--- CÂU TRẢ LỜI CỦA SINH VIÊN ---\n{student_answer_raw}\n\n"
        f"--- HÃY CHẤM ĐIỂM DƯỚI DẠNG JSON, KHÔNG THÊM CHỮ NÀO NGOÀI JSON ---"
    )

    try:
        resp = await asyncio.to_thread(
            openai_client.chat.completions.create,
            model=model_name,
            response_format={"type": "json_object"},
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ],
        )
        data = json.loads(resp.choices[0].message.content)
        score = min(max(float(data.get("diem_so", 0.0)), 0.0), max_score)
        feedback = data.get("phan_hoi", "Không có phản hồi.")
        return score, feedback
    except Exception as e:
        logger.error(f"Lỗi khi chấm điểm OpenAI: {e}")
        return 0.0, f"Lỗi hệ thống chấm điểm AI: {e}"

def convert_webm_to_wav(webm_path: str) -> Optional[str]:
    wav_path = webm_path.replace(".webm", ".wav")
    try:
        command = [
            "ffmpeg",
            "-hide_banner", "-loglevel", "warning", "-nostdin",
            "-fflags", "+genpts",
            "-i", webm_path,
            "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            "-y", wav_path,
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
        logger.info(f"Đã chuyển đổi thành công {webm_path} sang {wav_path}")
        return wav_path
    except subprocess.CalledProcessError as e:
        logger.error(f"Lỗi khi chạy ffmpeg: {e.stderr}")
        return None
    except FileNotFoundError:
        logger.error("Lỗi: không tìm thấy 'ffmpeg' trong PATH.")
        return None

# ====== WORKER CLASS ======
class Command(BaseCommand):
    help = 'Chạy worker lắng nghe các tác vụ AI từ channel layer'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.channel_layer = get_channel_layer()
        self.audio_chunks = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if torch.cuda.is_available() else "int8"
        self.stdout.write(f"Sử dụng thiết bị: {self.device} với kiểu tính toán {self.compute_type}")

        # PhoBERT
        self.stdout.write("Đang tải PhoBERT model...")
        try:
            self.phobert_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
            self.phobert_model = AutoModel.from_pretrained("vinai/phobert-base").to(self.device)
            self.phobert_model.eval()
            self.stdout.write("✅ PhoBERT đã sẵn sàng.")
        except Exception as e:
            self.phobert_model = None
            self.phobert_tokenizer = None
            self.stderr.write(self.style.ERROR(f"Lỗi khi tải PhoBERT: {e} - sẽ bỏ qua chấm điểm barem."))

        # OpenAI
        self.stdout.write("Đang cấu hình OpenAI client...")
        try:
            self.openai_client = openai.OpenAI()
            self.openai_client.models.list()
            self.stdout.write("✅ OpenAI (ChatGPT & Whisper) đã sẵn sàng.")
        except Exception as e:
            self.openai_client = None
            self.stderr.write(self.style.ERROR(f"Lỗi cấu hình OpenAI: {e} - đặt OPENAI_API_KEY trước."))

    @sync_to_async
    def get_question_and_barem(self, question_id):
        try:
            question_obj = Question.objects.select_related('subject').get(pk=question_id)
            quiz_file_path = os.path.join(settings.BASE_DIR, 'quiz_data', question_obj.subject.quiz_data_file)
            with open(quiz_file_path, 'r', encoding='utf-8') as f:
                all_barems = json.load(f)
            for barem in all_barems:
                if barem['id'] == question_obj.question_id_in_barem:
                    # gắn thêm text câu hỏi vào barem cho thuận tiện
                    barem['question'] = barem.get('question') or question_obj.question_text
                    return barem
        except Exception as e:
            logger.error(f"Không thể tải barem cho câu hỏi {question_id}: {e}")
        return None

    async def process_audio_and_transcribe(self, reply_channel, chunks, whisper_prompt: Optional[str] = None):
        if not chunks:
            logger.warning("Không có chunk âm thanh nào để xử lý.")
            return None

        # kiểm tra EBML header
        first = chunks[0]
        if len(first) < 4 or not has_ebml_header(first[:4]):
            logger.error("Các chunk không chứa EBML header. Dữ liệu audio không hợp lệ.")
            await self.channel_layer.send(reply_channel, {'type': 'exam.error', 'message': 'Dữ liệu audio không hợp lệ (thiếu header).'})
            return None

        unique_id = re.sub(r'[^a-zA-Z0-9]', '_', reply_channel)
        webm_path = os.path.join(settings.BASE_DIR, f'temp_audio_{unique_id}_{uuid4().hex[:8]}.webm')

        logger.info(f"Đang ghi {len(chunks)} chunk vào file {webm_path}...")
        try:
            with open(webm_path, 'wb') as f:
                for c in chunks:
                    f.write(c)
                f.flush()
                os.fsync(f.fileno())
        except Exception as e:
            logger.error(f"Lỗi ghi file tạm: {e}")
            await self.channel_layer.send(reply_channel, {'type': 'exam.error', 'message': 'Lỗi hệ thống khi ghi file âm thanh.'})
            return None

        wav_path = await asyncio.to_thread(convert_webm_to_wav, webm_path)
        if os.path.exists(webm_path):
            os.remove(webm_path)
        if not wav_path:
            await self.channel_layer.send(reply_channel, {'type': 'exam.error', 'message': 'Lỗi xử lý file âm thanh.'})
            return None

        duration, rms = wav_duration_and_rms(wav_path)
        logger.info(f"WAV duration ~ {duration:.2f}s; RMS ~ {rms:.1f}")
        if duration < 2.0 or rms < 50:
            msg = 'Âm thanh quá ngắn hoặc tín hiệu quá nhỏ. Vui lòng nói rõ ràng trong ít nhất 3 giây.'
            logger.warning(f"Từ chối xử lý audio: {msg} (Duration: {duration:.2f}s, RMS: {rms:.1f})")
            await self.channel_layer.send(reply_channel, {'type': 'exam.error', 'message': msg})
            if os.path.exists(wav_path):
                os.remove(wav_path)
            return ""

        logger.info(f"Đang gửi file âm thanh đến OpenAI Whisper API: {wav_path}...")
        try:
            with open(wav_path, "rb") as audio_file:
                kwargs = dict(model="whisper-1", file=audio_file, language="vi", temperature=0)
                if whisper_prompt:
                    kwargs["prompt"] = whisper_prompt
                transcription = await asyncio.to_thread(self.openai_client.audio.transcriptions.create, **kwargs)
            raw = transcription.text.strip()
            logger.info(f"Transcript nhận được: '{raw}'")
            return raw
        except Exception as e:
            logger.error(f"Lỗi Whisper: {e}")
            return ""
        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)

    async def process_main_question(self, message):
        reply_channel = message['reply_channel']
        question_id = message.get('question_id')
        session_id = message.get('session_id')
        chunks = message.get('__chunks', [])
        if not all([session_id, question_id, chunks]):
            logger.error(f"Tác vụ 'main' thiếu dữ liệu. session_id={session_id}, question_id={question_id}, chunks={len(chunks)}")
            return
        try:
            question_barem = await self.get_question_and_barem(question_id)
            if not question_barem:
                await self.channel_layer.send(reply_channel, {'type': 'exam.error', 'message': 'Lỗi hệ thống: Không tìm thấy barem.'})
                return

            whisper_prompt = f"Bài thi vấn đáp. Câu hỏi: {question_barem.get('question','')}"
            raw_transcript = await self.process_audio_and_transcribe(reply_channel, chunks, whisper_prompt=whisper_prompt)
            if raw_transcript is None or raw_transcript == "":
                return

            rephrased = await rephrase_text_with_chatgpt(raw_transcript, question_barem['question'], self.openai_client)

            # chấm hybrid
            traditional_task = asyncio.to_thread(
                score_student_answer_traditional,
                rephrased, question_barem, self.phobert_tokenizer, self.phobert_model, self.device
            )
            openai_task = score_student_answer_with_openai(rephrased, question_barem, self.openai_client)
            traditional_score, (openai_score, feedback) = await asyncio.gather(traditional_task, openai_task)

            WEIGHT_OPENAI = 0.7
            WEIGHT_TRADITIONAL = 0.3
            final_score = (openai_score * WEIGHT_OPENAI) + (traditional_score * WEIGHT_TRADITIONAL)
            final_score = float(min(max(final_score, 0.0), 10.0))

            logger.info(f"Chấm điểm Q{question_id}: PhoBERT={traditional_score:.2f}, OpenAI={openai_score:.2f} -> Final={final_score:.2f}")

            # lưu DB
            try:
                session = await sync_to_async(ExamSession.objects.get)(pk=session_id)
                question = await sync_to_async(Question.objects.get)(pk=question_id)
            except ObjectDoesNotExist as e:
                logger.error(f"Không tìm thấy ExamSession/Question: {e}")
                return

            exam_result = await sync_to_async(ExamResult.objects.create)(
                session=session, question=question, transcript=rephrased, score=final_score, feedback=feedback
            )

            # gửi về client
            await self.channel_layer.send(reply_channel, {
                'type': 'exam.result',
                'message': {
                    'type': 'main_question_complete',
                    'data': {
                        'result_id': exam_result.id,
                        'score': exam_result.score,
                        'question_id': question_id,
                        'transcript': rephrased,
                        'feedback': feedback
                    }
                }
            })
        except Exception as e:
            logger.error(f"Lỗi không mong muốn trong process_main_question: {e}", exc_info=True)
            await self.channel_layer.send(reply_channel, {'type': 'exam.error', 'message': f'Lỗi worker: {str(e)}'})

    async def process_supplementary_question(self, message):
        reply_channel = message['reply_channel']
        session_id = message.get('session_id')
        question_text = message.get('question_text')
        max_score = message.get('max_score')
        chunks = message.get('__chunks', [])

        if not all([session_id, question_text, max_score is not None, chunks]):
            logger.error(f"Tác vụ 'supplementary' thiếu dữ liệu.")
            return
        try:
            raw_transcript = await self.process_audio_and_transcribe(
                reply_channel, chunks, whisper_prompt=f"Bài thi vấn đáp. Câu hỏi phụ: {question_text}"
            )
            if raw_transcript is None or raw_transcript == "":
                return

            rephrased = await rephrase_text_with_chatgpt(raw_transcript, question_text, self.openai_client)

            # tạo barem giả cho LLM (chỉ có max_score, không KP)
            pseudo_barem = {"question": question_text, "key_points": [], "max_score": float(max_score)}
            openai_score, feedback = await score_student_answer_with_openai(
                rephrased, pseudo_barem, self.openai_client
            )
            openai_score = float(min(max(openai_score, 0.0), float(max_score)))

            session = await sync_to_async(ExamSession.objects.get)(pk=session_id)
            supp = await sync_to_async(SupplementaryResult.objects.create)(
                session=session, question_text=question_text, transcript=rephrased,
                score=openai_score, max_score=max_score, feedback=feedback
            )

            await self.channel_layer.send(reply_channel, {
                'type': 'exam.result',
                'message': {
                    'type': 'supp_question_complete',
                    'data': {
                        'result_id': supp.id,
                        'score': supp.score,
                        'max_score': supp.max_score,
                        'question_text': question_text,
                        'transcript': rephrased,
                        'feedback': feedback
                    }
                }
            })
        except Exception as e:
            logger.error(f"Lỗi không mong muốn trong process_supplementary_question: {e}", exc_info=True)
            await self.channel_layer.send(reply_channel, {'type': 'exam.error', 'message': f'Lỗi worker: {str(e)}'})

    async def run(self):
        logger.info("Worker đang lắng nghe trên kênh 'asr-tasks'...")
        while True:
            message = await self.channel_layer.receive('asr-tasks')
            task_type = message.get('type')
            reply_channel = message.get('reply_channel')
            if not reply_channel:
                continue

            if task_type == 'asr.stream.start':
                logger.info(f"Bắt đầu stream cho kênh {reply_channel}")
                self.audio_chunks[reply_channel] = []

            elif task_type == 'asr.chunk':
                if reply_channel in self.audio_chunks:
                    chunk_data = message.get('audio_chunk', b'')
                    if chunk_data:
                        self.audio_chunks[reply_channel].append(chunk_data)

            elif task_type == 'asr.stream.end':
                # Đọc mode / process_type (fallback)
                mode = (message.get('mode')
                        or message.get('process_type')
                        or ('main' if message.get('question_id') else 'supplementary'))
                logger.info(f"Kết thúc stream, nhận lệnh xử lý '{mode}' cho kênh {reply_channel}")

                chunks = self.audio_chunks.pop(reply_channel, [])
                message['__chunks'] = chunks
                if not chunks:
                    logger.warning(f"Không có chunk audio nào để xử lý cho kênh {reply_channel}. Bỏ qua.")
                    continue

                if mode in ('main', 'primary'):
                    asyncio.create_task(self.process_main_question(message))
                else:
                    asyncio.create_task(self.process_supplementary_question(message))

            else:
                logger.warning(f"Bỏ qua message không hỗ trợ: {task_type}")

    def handle(self, *args, **options):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.stdout.write(self.style.SUCCESS('Starting AI Worker...'))
        try:
            asyncio.run(self.run())
        except KeyboardInterrupt:
            self.stdout.write(self.style.WARNING('Worker stopped by user.'))
