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
import openai

from django.core.management.base import BaseCommand
from django.conf import settings
from channels.layers import get_channel_layer
from asgiref.sync import sync_to_async
from qna.models import Question, ExamSession, ExamResult

logger = logging.getLogger(__name__)


# --- CÁC HÀM XỬ LÝ NLP VÀ AI ---

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


async def rephrase_text_with_chatgpt(text, question_barem, client):
    if not text or client is None:
        return text

    barem_text = "\n".join([f"- {kp['text']}" for kp in question_barem['key_points']])
    system_prompt = "Bạn là một trợ lý ngôn ngữ chuyên sửa lỗi chính tả tiếng Việt một cách chính xác."
    user_prompt = f"""Văn bản sau là câu trả lời của sinh viên, được nhận dạng từ giọng nói. Nhiệm vụ của bạn là chỉ sửa các lỗi chính tả (ví dụ: 'chủng' thành 'chuẩn') do giọng nói vùng miền hoặc phát âm sai.
Tuyệt đối KHÔNG được loại bỏ các từ bị lặp (ví dụ: 'là là là' phải giữ nguyên là 'là là là'), không thêm từ, và không thay đổi cấu trúc hay trật tự câu gốc của sinh viên.
Trả về duy nhất văn bản đã được sửa.

--- CÂU HỎI ---
{question_barem['question']}

--- BAREM ĐIỂM THAM KHẢO ---
{barem_text}

--- CÂU TRẢ LỜI CỦA SINH VIÊN (VĂN BẢN GỐC TỪ ASR) ---
{text}
"""
    try:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Lỗi khi rephrase văn bản bằng OpenAI: {e}")
        return text


async def score_with_chatgpt(student_answer, question_text, barem_key_points, client):
    system_prompt = """Bạn là một chuyên gia chấm điểm câu trả lời vấn đáp. Nhiệm vụ của bạn là chấm điểm câu trả lời của sinh viên dựa trên barem điểm đã cho một cách nghiêm ngặt.
Đầu ra của bạn PHẢI là một chuỗi JSON hợp lệ và chỉ chứa duy nhất JSON đó, không giải thích gì thêm.
Cấu trúc JSON bắt buộc: {"diem_so": float, "phan_hoi": "string", "phan_tich": [{"text": "string", "matched": boolean}]}.
"""
    prompt_parts = [
        f"--- CÂU HỎI ---",
        f"Câu hỏi: {question_text}",
        f"--- BAREM ĐIỂM ---"
    ]
    for kp in barem_key_points:
        prompt_parts.append(f"- {kp['text']} (Trọng số: {kp['weight']:.1f})")

    prompt_parts.extend([
        "--- CÂU TRẢ LỜI CỦA SINH VIÊN ---",
        student_answer,
        "--- HÃY CHẤM ĐIỂM VÀ ĐƯA RA KẾT QUẢ DƯỚI DẠNG JSON NHƯ ĐÃ YÊU CẦU ---"
    ])
    user_prompt = "\n".join(prompt_parts)

    try:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        data = json.loads(response.choices[0].message.content)
        return {"score": float(data.get("diem_so", 0.0)), "feedback": data.get("phan_hoi", "Không có phản hồi."),
                "analysis": data.get("phan_tich", [])}
    except Exception as e:
        logger.error(f"Lỗi ChatGPT: {e}")
        return {"score": 0.0, "feedback": "Lỗi hệ thống chấm điểm AI.", "analysis": []}


async def generate_follow_up_question(client, question_text, missed_key_points, student_answer):
    system_prompt = "Bạn là một trợ giảng AI chuyên ra câu hỏi vấn đáp. Nhiệm vụ của bạn là tạo ra một câu hỏi phụ ngắn gọn, tập trung vào một điểm yếu duy nhất trong câu trả lời của sinh viên dựa trên barem."

    # Chuyển danh sách các ý bị lỡ thành một chuỗi
    missed_points_text = "\n".join([f"- {kp['text']}" for kp in missed_key_points])

    user_prompt = f"""Dựa vào các thông tin dưới đây, hãy tạo ra MỘT câu hỏi phụ ngắn gọn để kiểm tra lại kiến thức mà sinh viên đã trả lời sai hoặc còn thiếu.
Câu hỏi phải rõ ràng, dễ hiểu và chỉ tập trung vào MỘT trong các ý mà sinh viên đã bỏ lỡ.

--- CÂU HỎI CHÍNH ---
{question_text}

--- CÁC Ý CHÍNH TRONG BAREM MÀ SINH VIÊN ĐÃ BỎ LỠ ---
{missed_points_text}

--- CÂU TRẢ LỜI CỦA SINH VIÊN ---
"{student_answer}"

--- YÊU CẦU ---
Hãy tạo MỘT câu hỏi phụ duy nhất.
"""
    try:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.5,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Lỗi khi tạo câu hỏi phụ bằng OpenAI: {e}")
        # Câu hỏi dự phòng trong trường hợp API lỗi
        return "Bạn có thể giải thích rõ hơn về một trong những ý bạn vừa trình bày không?"


async def score_follow_up_question(client, follow_up_question, follow_up_answer):
    system_prompt = "Bạn là một giám khảo AI. Hãy chấm điểm câu trả lời cho câu hỏi phụ trên thang điểm 2.0. Trả lời bằng một chuỗi JSON hợp lệ có dạng: {\"diem_phu\": float, \"nhan_xet_phu\": \"string\"}."
    user_prompt = f"""--- CÂU HỎI PHỤ ---
{follow_up_question}

--- CÂU TRẢ LỜI CỦA SINH VIÊN ---
{follow_up_answer}

--- JSON KẾT QUẢ (thang điểm 2.0) ---"""
    try:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        data = json.loads(response.choices[0].message.content)
        score = min(float(data.get("diem_phu", 0.0)), 2.0)
        return score, data.get("nhan_xet_phu", "")
    except Exception as e:
        logger.error(f"Lỗi khi chấm điểm câu hỏi phụ: {e}")
        return 0.0, "Lỗi hệ thống khi chấm câu hỏi phụ."


def convert_webm_to_wav(webm_path):
    wav_path = webm_path.replace(".webm", ".wav")
    try:
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

        self.stdout.write("Đang tải PhoBERT model...")
        self.phobert_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
        self.phobert_model = AutoModel.from_pretrained("vinai/phobert-base").to(self.device)
        self.phobert_model.eval()
        self.stdout.write("✅ PhoBERT đã sẵn sàng.")

        self.stdout.write("Đang cấu hình mô hình ChatGPT (OpenAI)...")
        try:
            self.openai_client = openai.OpenAI()
            self.openai_client.models.list()
            self.stdout.write("✅ OpenAI (ChatGPT & Whisper) đã sẵn sàng.")
        except Exception as e:
            self.openai_client = None
            self.stderr.write(self.style.ERROR(
                f"Lỗi cấu hình OpenAI: {e} - Hãy chắc chắn bạn đã đặt biến môi trường OPENAI_API_KEY."))

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

    @sync_to_async
    def get_exam_result_and_barem(self, result_id):
        try:
            result = ExamResult.objects.select_related('question', 'question__subject').get(pk=result_id)
            subject = result.question.subject
            quiz_file_path = os.path.join(settings.BASE_DIR, 'quiz_data', subject.quiz_data_file)
            with open(quiz_file_path, 'r', encoding='utf-8') as f:
                all_barems = json.load(f)
            for barem in all_barems:
                if barem['id'] == result.question.question_id_in_barem:
                    return result, barem
        except Exception as e:
            logger.error(f"Không thể tải barem cho result_id {result_id}: {e}")
        return None, None

    async def process_audio_and_transcribe(self, reply_channel, chunks):
        if not chunks:
            logger.warning("Không có chunk âm thanh nào để xử lý.")
            return None

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
            await self.channel_layer.send(reply_channel, {'type': 'exam.result', 'message': {'type': 'error', 'data': {
                'message': 'Lỗi xử lý file âm thanh. Vui lòng thử lại.'}}})
            return None

        logger.info(f"Đang gửi file âm thanh đến OpenAI Whisper API: {wav_path}...")
        try:
            with open(wav_path, "rb") as audio_file:
                transcription = await asyncio.to_thread(
                    self.openai_client.audio.transcriptions.create,
                    model="whisper-1",
                    file=audio_file,
                    language="vi"
                )
            raw_transcript = transcription.text.strip()
            logger.info(f"Transcript nhận được: '{raw_transcript}'")
            return raw_transcript
        except Exception as e:
            logger.error(f"Lỗi khi nhận dạng giọng nói với OpenAI API: {e}")
            return ""
        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)

    async def process_end(self, message):
        reply_channel = message['reply_channel']
        question_id = message['question_id']
        session_id = message.get('session_id')
        chunks = self.audio_chunks.pop(reply_channel, [])

        raw_transcript = await self.process_audio_and_transcribe(reply_channel, chunks)
        if raw_transcript is None: return

        if not raw_transcript:
            await self.channel_layer.send(reply_channel, {'type': 'exam.result', 'message': {'type': 'error', 'data': {
                'message': 'Không nhận dạng được giọng nói. Vui lòng thử lại.'}}})
            return

        question_barem = await self.get_question_and_barem(question_id)
        if not question_barem:
            await self.channel_layer.send(reply_channel, {'type': 'exam.result', 'message': {'type': 'error', 'data': {
                'message': 'Lỗi hệ thống: Không tìm thấy barem chấm điểm.'}}})
            return

        rephrased_transcript = await rephrase_text_with_chatgpt(raw_transcript, question_barem, self.openai_client)

        chatgpt_result = await score_with_chatgpt(rephrased_transcript, question_barem['question'],
                                                  question_barem['key_points'], self.openai_client)
        score_lan_1 = chatgpt_result.get('score', 0.0)

        session = await sync_to_async(ExamSession.objects.get)(pk=session_id)
        question = await sync_to_async(Question.objects.get)(pk=question_id)
        exam_result = await sync_to_async(ExamResult.objects.create)(
            session=session,
            question=question,
            transcript=rephrased_transcript,
            score=score_lan_1,
            feedback=chatgpt_result.get('feedback'),
            analysis=chatgpt_result.get('analysis')
        )

        if score_lan_1 < 5.0:
            # Xác định các ý bị lỡ từ barem
            analysis = chatgpt_result.get('analysis', [])
            missed_key_points = [item for item in analysis if not item.get('matched')]

            if not missed_key_points:  # Nếu không có ý nào bị lỡ rõ ràng, dùng barem đầy đủ
                missed_key_points = question_barem['key_points']

            follow_up_q = await generate_follow_up_question(self.openai_client, question_barem['question'],
                                                            missed_key_points, rephrased_transcript)

            exam_result.follow_up_question = follow_up_q
            await sync_to_async(exam_result.save)()

            await self.channel_layer.send(reply_channel, {'type': 'exam.result', 'message': {
                'type': 'follow_up_required',
                'data': {
                    'exam_result_id': exam_result.id,
                    'score_1': score_lan_1,
                    'feedback_1': exam_result.feedback,
                    'follow_up_question': follow_up_q
                }
            }})
        else:
            await self.channel_layer.send(reply_channel, {'type': 'exam.result', 'message': {
                'type': 'complete',
                'data': {
                    'exam_result_id': exam_result.id,
                    'transcript': exam_result.transcript,
                    'final_score': exam_result.score,
                    'feedback': exam_result.feedback,
                    'analysis': exam_result.analysis
                }
            }})

    async def process_follow_up(self, message):
        reply_channel = message['reply_channel']
        exam_result_id = message['exam_result_id']
        chunks = self.audio_chunks.pop(reply_channel, [])

        raw_transcript = await self.process_audio_and_transcribe(reply_channel, chunks)
        if raw_transcript is None: return

        exam_result, barem = await self.get_exam_result_and_barem(exam_result_id)
        if not exam_result:
            return

        rephrased_transcript = await rephrase_text_with_chatgpt(raw_transcript, barem, self.openai_client)

        follow_up_score, follow_up_feedback = await score_follow_up_question(self.openai_client,
                                                                             exam_result.follow_up_question,
                                                                             rephrased_transcript)

        exam_result.follow_up_transcript = rephrased_transcript
        exam_result.follow_up_score = follow_up_score
        exam_result.follow_up_feedback = follow_up_feedback
        await sync_to_async(exam_result.save)()

        final_combined_score = exam_result.score + exam_result.follow_up_score

        await self.channel_layer.send(reply_channel, {'type': 'exam.result', 'message': {
            'type': 'complete',
            'data': {
                'exam_result_id': exam_result.id,
                'transcript': exam_result.transcript,
                'final_score': final_combined_score,
                'feedback': exam_result.feedback,
                'analysis': exam_result.analysis,
                'follow_up_question': exam_result.follow_up_question,
                'follow_up_transcript': exam_result.follow_up_transcript,
                'follow_up_score': exam_result.follow_up_score,
                'follow_up_feedback': exam_result.follow_up_feedback,
            }
        }})

    async def run(self):
        logger.info("Worker đang lắng nghe trên kênh 'asr-tasks'...")
        while True:
            message = await self.channel_layer.receive('asr-tasks')
            task_type = message.get('type')

            reply_channel = message.get('reply_channel')
            if not reply_channel: continue

            if 'chunk' in task_type:
                if reply_channel not in self.audio_chunks:
                    self.audio_chunks[reply_channel] = []
                self.audio_chunks[reply_channel].append(message['audio_chunk'])

            elif task_type == 'asr.process.end':
                asyncio.create_task(self.process_end(message))

            elif task_type == 'asr.process.follow_up':
                asyncio.create_task(self.process_follow_up(message))

    def handle(self, *args, **options):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.stdout.write(self.style.SUCCESS('Starting AI Worker...'))
        try:
            asyncio.run(self.run())
        except KeyboardInterrupt:
            self.stdout.write(self.style.WARNING('Worker stopped by user.'))