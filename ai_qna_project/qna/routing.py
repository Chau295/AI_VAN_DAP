# qna/routing.py
from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    re_path(r'^ws/exam/(?P<subject_code>\w+)/$', consumers.ExamConsumer.as_asgi()),
]