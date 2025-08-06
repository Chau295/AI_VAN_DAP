# ai_qna_project/urls.py
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
# THÊM DÒNG IMPORT NÀY
from django.contrib.staticfiles.urls import staticfiles_urlpatterns

urlpatterns = [
    path('admin/', admin.site.urls),
    path('accounts/', include('django.contrib.auth.urls')),
    path('', include('qna.urls')),
]

# Cấu hình để phục vụ tệp media do người dùng tải lên (bạn đã có)
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

# THÊM ĐOẠN MÃ NÀY ĐỂ PHỤC VỤ TỆP STATIC KHI CHẠY BẰNG DAPHNE
urlpatterns += staticfiles_urlpatterns()