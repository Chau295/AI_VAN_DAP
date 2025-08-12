# qna/consumers.py
import json
import logging
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.layers import get_channel_layer
from asgiref.sync import sync_to_async
from .models import Question, ExamResult

logger = logging.getLogger(__name__)


class ExamConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.user = self.scope["user"]
        if not self.user.is_authenticated:
            await self.close()
            return

        query_string = self.scope['query_string'].decode()
        params = dict(q.split('=') for q in query_string.split('&') if '=' in q)

        self.question_id = params.get('question_id')
        # Lấy exam_result_id cho các câu hỏi phụ
        self.exam_result_id = params.get('exam_result_id')

        # Logic xác thực: Phải có question_id hoặc exam_result_id
        if not self.question_id and not self.exam_result_id:
            logger.error(f"User {self.user.username} connected with no question_id or exam_result_id.")
            await self.close()
            return

        # Nếu là câu hỏi phụ, xác thực exam_result_id
        if self.exam_result_id and not await self.is_valid_exam_result(self.exam_result_id):
            logger.error(
                f"User {self.user.username} tried to connect with invalid exam_result_id: {self.exam_result_id}")
            await self.close()
            return

        self.subject_code = self.scope['url_route']['kwargs']['subject_code']
        self.room_group_name = f'exam_{self.user.id}_{self.subject_code}'

        await self.channel_layer.group_add(self.room_group_name, self.channel_name)
        await self.accept()
        logger.info(
            f"User {self.user.username} connected for exam {self.subject_code}. Q_ID: {self.question_id}, Result_ID: {self.exam_result_id}")

    async def disconnect(self, close_code):
        if hasattr(self, 'room_group_name'):
            await self.channel_layer.group_discard(self.room_group_name, self.channel_name)
        logger.info(f"User {self.user.username} disconnected.")

    async def receive(self, text_data=None, bytes_data=None):
        channel_layer = get_channel_layer()

        if text_data:
            try:
                data = json.loads(text_data)
                if data.get('type') == 'end_of_stream':
                    logger.info(f"Received end_of_stream from {self.user.username}.")

                    # Phân biệt tác vụ cho câu hỏi chính và câu hỏi phụ
                    if self.exam_result_id:
                        # Đây là câu trả lời cho câu hỏi phụ
                        await channel_layer.send('asr-tasks', {
                            'type': 'asr.process.follow_up',
                            'reply_channel': self.channel_name,
                            'exam_result_id': self.exam_result_id
                        })
                    else:
                        # Đây là câu trả lời cho câu hỏi chính
                        await channel_layer.send('asr-tasks', {
                            'type': 'asr.process.end',
                            'reply_channel': self.channel_name,
                            'question_id': self.question_id,
                            'session_id': data.get('session_id')
                        })
                    return
            except json.JSONDecodeError:
                pass

        if bytes_data:
            # Gửi chunk âm thanh đến worker
            task_type = 'asr.process.follow_up_chunk' if self.exam_result_id else 'asr.process.chunk'
            await channel_layer.send('asr-tasks', {
                'type': task_type,
                'audio_chunk': bytes_data,
                'reply_channel': self.channel_name
            })

    async def exam_result(self, event):
        # Gửi kết quả cuối cùng hoặc yêu cầu câu hỏi phụ về client
        await self.send(text_data=json.dumps(event['message']))
        logger.info(f"Sent result to {self.user.username}. Status: {event['message'].get('type')}")

    @sync_to_async
    def is_valid_question(self, question_id):
        return Question.objects.filter(pk=question_id).exists()

    @sync_to_async
    def is_valid_exam_result(self, result_id):
        # Đảm bảo ExamResult tồn tại và thuộc về user hiện tại
        return ExamResult.objects.filter(pk=result_id, session__user=self.user).exists()