from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.contrib.auth.models import User
from .models import (
    Subject, Question, ExamSession, ExamResult,
    SupplementaryResult, UserProfile
)


# Định nghĩa một inline admin cho UserProfile
class UserProfileInline(admin.StackedInline):
    model = UserProfile
    can_delete = False
    verbose_name_plural = 'Hồ sơ sinh viên'
    fk_name = 'user'


# Định nghĩa một User admin mới để bao gồm UserProfile
class UserAdmin(BaseUserAdmin):
    inlines = (UserProfileInline,)


# Hủy đăng ký User admin mặc định và đăng ký lại với phiên bản mới của chúng ta
admin.site.unregister(User)
admin.site.register(User, UserAdmin)

@admin.register(Subject)
class SubjectAdmin(admin.ModelAdmin):
    list_display = ('name', 'subject_code')
    search_fields = ('name', 'subject_code')

@admin.register(Question)
class QuestionAdmin(admin.ModelAdmin):
    list_display = ('question_text', 'subject', 'question_id_in_barem', 'is_supplementary')
    list_filter = ('subject', 'is_supplementary')
    search_fields = ('question_text', 'subject__name')

@admin.register(ExamSession)
class ExamSessionAdmin(admin.ModelAdmin):
    list_display = (
        'user',
        'subject',
        'created_at',
        'is_completed',
        'final_score'  # ĐÃ THAY ĐỔI
    )
    list_filter = ('subject', 'is_completed', 'user')
    search_fields = ('user__username', 'subject__name')
    date_hierarchy = 'created_at'
    filter_horizontal = ('questions',)

@admin.register(ExamResult)
class ExamResultAdmin(admin.ModelAdmin):
    list_display = ('session', 'question', 'score', 'answered_at')
    list_filter = ('session__subject',)
    search_fields = ('session__user__username', 'question__question_text')

@admin.register(SupplementaryResult)
class SupplementaryResultAdmin(admin.ModelAdmin):
    list_display = ('session', 'question_text', 'score', 'max_score', 'created_at')
    list_filter = ('session__subject',)
    search_fields = ('session__user__username', 'question_text')
