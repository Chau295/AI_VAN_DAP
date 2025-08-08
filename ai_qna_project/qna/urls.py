# qna/urls.py
from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.dashboard_view, name='dashboard'),
    path('profile/', views.profile_view, name='profile'),
    path('exam/<str:subject_code>/', views.exam_view, name='exam_page'),
    path('register/', views.register_view, name='register'),

    # URLs for APIs
    path('update_image/', views.update_profile_image, name='update_profile_image'),
    path('save_result/', views.save_exam_result, name='save_result'),
    path('complete_session/', views.complete_exam_session, name='complete_session'),

    # Cấu trúc URL Lịch sử mới
    path('history/', views.history_view, name='history'),
    path('history/session/<int:session_id>/', views.history_detail_view, name='history_detail'),
    path('history/result/<int:result_id>/', views.exam_result_detail_view, name='exam_result_detail'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)