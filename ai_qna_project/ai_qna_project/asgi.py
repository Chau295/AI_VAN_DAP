# ai_qna_project/asgi.py (ĐÃ SỬA LỖI)
import os
from django.core.asgi import get_asgi_application
import django

# THÊM DÒNG NÀY ĐỂ KHỞI TẠO DJANGO TRƯỚC
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ai_qna_project.settings')
django.setup()

# SAU ĐÓ MỚI IMPORT CÁC THÀNH PHẦN CỦA CHANNELS
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
import qna.routing

# application được định nghĩa sau khi django.setup() đã được gọi
application = ProtocolTypeRouter({
  "http": get_asgi_application(),
  "websocket": AuthMiddlewareStack(
        URLRouter(
            qna.routing.websocket_urlpatterns
        )
    ),
})