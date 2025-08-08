# ai_qna_project/qna/admin.py
from django.contrib import admin
from .models import UserProfile, Subject, Question, ExamResult, ExamSession

@admin.register(Subject)
class SubjectAdmin(admin.ModelAdmin):
    list_display = ('name', 'subject_code', 'quiz_data_file')

@admin.register(Question)
class QuestionAdmin(admin.ModelAdmin):
    list_display = ('question_text', 'subject', 'question_id_in_barem')
    list_filter = ('subject',)

@admin.register(ExamSession)
class ExamSessionAdmin(admin.ModelAdmin):
    list_display = ('user', 'subject', 'created_at', 'is_completed')
    list_filter = ('subject', 'user', 'is_completed')

@admin.register(ExamResult)
class ExamResultAdmin(admin.ModelAdmin):
    # SỬA LỖI Ở ĐÂY: Dùng '__' để truy cập các trường của model liên quan
    list_display = ('get_user', 'get_subject', 'score', 'answered_at')
    list_filter = ('session__subject', 'session__user')
    search_fields = ('session__user__username', 'session__subject__name')

    # Định nghĩa các phương thức để lấy thông tin từ session
    @admin.display(description='Sinh viên', ordering='session__user')
    def get_user(self, obj):
        return obj.session.user

    @admin.display(description='Môn học', ordering='session__subject')
    def get_subject(self, obj):
        return obj.session.subject

admin.site.register(UserProfile)