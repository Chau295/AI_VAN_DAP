# ai_qna_project/asgi.py (ĐÃ SỬA LỖI)
import os
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
import qna.routing  # Sửa lại thành cách import trực tiếp và đơn giản

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ai_qna_project.settings')

application = ProtocolTypeRouter({
  "http": get_asgi_application(),
  "websocket": AuthMiddlewareStack(
        URLRouter(
            qna.routing.websocket_urlpatterns
        )
    ),
})