# qna/consumers.py
import json
import logging
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.layers import get_channel_layer
from asgiref.sync import sync_to_async
from .models import Question

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

        if not self.question_id or not await self.is_valid_question(self.question_id):
            logger.error(f"User {self.user.username} tried to connect with invalid question_id: {self.question_id}")
            await self.close()
            return

        self.subject_code = self.scope['url_route']['kwargs']['subject_code']
        self.room_group_name = f'exam_{self.user.id}_{self.subject_code}'

        await self.channel_layer.group_add(self.room_group_name, self.channel_name)
        await self.accept()
        logger.info(f"User {self.user.username} connected for exam {self.subject_code}, question {self.question_id}.")

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
                    logger.info(f"Received end_of_stream from {self.user.username} for question {self.question_id}.")
                    await channel_layer.send('asr-tasks', {
                        'type': 'asr.process.end',
                        'reply_channel': self.channel_name,
                        'question_id': self.question_id
                    })
                    return
            except json.JSONDecodeError:
                pass

        if bytes_data:
            await channel_layer.send('asr-tasks', {
                'type': 'asr.process.chunk',
                'audio_chunk': bytes_data,
                'reply_channel': self.channel_name
            })

    async def exam_result(self, event):
        await self.send(text_data=json.dumps({
            'type': 'result',
            'data': event['message']
        }))
        logger.info(f"Sent final result to {self.user.username}.")

    async def partial_transcript(self, event):
        await self.send(text_data=json.dumps({
            'type': 'partial_transcript',
            'data': event['message']
        }))

    @sync_to_async
    def is_valid_question(self, question_id):
        return Question.objects.filter(pk=question_id).exists()