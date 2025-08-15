# ai_qna_project/qna/models.py
from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

class Conversation(models.Model):
    question_text = models.CharField(max_length=255)
    answer_text = models.TextField()
    timestamp = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return self.question_text

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    profile_image = models.ImageField(upload_to='profile_images/', null=True, blank=True)
    full_name = models.CharField(max_length=255, blank=True, null=True)
    student_id = models.CharField(max_length=50, unique=True, blank=True, null=True)
    class_name = models.CharField(max_length=100, blank=True, null=True)

    def __str__(self):
        return self.user.username

class Subject(models.Model):
    name = models.CharField(max_length=255, verbose_name="Tên môn học")
    subject_code = models.CharField(max_length=20, unique=True, verbose_name="Mã môn học")
    quiz_data_file = models.CharField(max_length=255, blank=True, null=True, help_text="Ví dụ: data_analysis_quiz.json")

    def __str__(self):
        return self.name

class Question(models.Model):
    subject = models.ForeignKey(Subject, on_delete=models.CASCADE, related_name="questions", verbose_name="Môn học")
    question_text = models.TextField(verbose_name="Nội dung câu hỏi")
    question_id_in_barem = models.CharField(max_length=20, verbose_name="ID câu hỏi trong tệp barem",
                                            help_text="Ví dụ: Q1, Q2...")
    is_supplementary = models.BooleanField(default=False, verbose_name="Là câu hỏi phụ")

    def __str__(self):
        return f"{self.subject.subject_code} - {self.question_text[:50]}..."

class ExamSession(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='exam_sessions')
    subject = models.ForeignKey(Subject, on_delete=models.CASCADE)
    questions = models.ManyToManyField(Question, related_name='exam_sessions')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="Ngày thi")
    is_completed = models.BooleanField(default=False)
    completed_at = models.DateTimeField(null=True, blank=True, verbose_name="Thời gian hoàn thành")
    final_score = models.FloatField(null=True, blank=True)

    def __str__(self):
        return f"Bài thi môn {self.subject.name} của {self.user.username} ngày {self.created_at.strftime('%d/%m/%Y')}"

class ExamResult(models.Model):
    session = models.ForeignKey(ExamSession, on_delete=models.CASCADE, related_name='results')
    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    transcript = models.TextField(verbose_name="Nội dung trả lời")
    score = models.FloatField(verbose_name="Điểm số")
    feedback = models.TextField(verbose_name="Nhận xét của AI", null=True, blank=True)
    analysis = models.JSONField(verbose_name="Phân tích chi tiết", null=True, blank=True)
    answered_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Kết quả câu hỏi {self.question.id} của {self.session.user.username}"

class SupplementaryResult(models.Model):
    """Lưu kết quả cho một câu hỏi phụ."""
    session = models.ForeignKey(ExamSession, on_delete=models.CASCADE, related_name='supplementary_results')
    question_text = models.TextField()
    transcript = models.TextField(blank=True, null=True)
    score = models.FloatField(default=0.0)
    max_score = models.FloatField(default=1.0)
    feedback = models.TextField(blank=True, null=True)
    analysis = models.JSONField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Supplementary result for {self.session.user.username} - Score: {self.score}/{self.max_score}"