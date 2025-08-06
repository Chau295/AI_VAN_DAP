# qna/admin.py
from django.contrib import admin
from .models import UserProfile, Subject, Question, ExamResult

@admin.register(Subject)
class SubjectAdmin(admin.ModelAdmin):
    list_display = ('name', 'subject_code', 'quiz_data_file')

@admin.register(Question)
class QuestionAdmin(admin.ModelAdmin):
    list_display = ('question_text', 'subject', 'question_id_in_barem')
    list_filter = ('subject',)

@admin.register(ExamResult)
class ExamResultAdmin(admin.ModelAdmin):
    list_display = ('user', 'subject', 'score', 'exam_date')
    list_filter = ('subject', 'user')
    search_fields = ('user__username', 'subject__name')

admin.site.register(UserProfile)