from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from django.views.generic import RedirectView   # ⬅️ THÊM DÒNG NÀY
from . import views

# (khuyến nghị) nếu bạn include với namespace: app_name = "qna"

urlpatterns = [
    # ⬅️ THÊM DÒNG NÀY: chuyển "/" về trang dashboard
    path("", RedirectView.as_view(pattern_name="dashboard", permanent=False), name="root"),

    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('exam_page/<str:subject_code>/', views.exam_view, name='exam_page'),

    path('profile/', views.profile_view, name='profile'),
    path('profile/update-image/', views.update_profile_image, name='update_profile_image'),

    path('history/', views.history_view, name='history'),
    path('history/<int:session_id>/', views.history_detail_view, name='history_detail'),

    # APIs
    path('api/save_exam_result/', views.save_exam_result, name='save_exam_result'),
    path('api/save_supplementary_result/', views.save_supplementary_result, name='save_supplementary_result'),
    path('api/get_supplementary/<int:session_id>/', views.get_supplementary_for_session,
         name='get_supplementary_for_session'),
    path('api/finalize_session/<int:session_id>/', views.finalize_session_view, name='finalize_session'),

    # Legacy
    path('api/supplementary/<int:session_id>/', views.get_supplementary_questions_api,
         name='get_supplementary_questions_api'),
    path('api/supplementary/random/', views.get_supplementary_questions, name='get_supplementary_questions'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
