# qna/models.py
from django.db import models
from django.utils import timezone
from django.contrib.auth.models import User

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
    question_id_in_barem = models.CharField(max_length=20, verbose_name="ID câu hỏi trong tệp barem", help_text="Ví dụ: Q1, Q2...")

    def __str__(self):
        return f"{self.subject.subject_code} - {self.question_text[:50]}..."

class ExamResult(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, verbose_name="Sinh viên")
    subject = models.ForeignKey(Subject, on_delete=models.CASCADE, verbose_name="Môn học")
    question = models.ForeignKey(Question, on_delete=models.CASCADE, verbose_name="Câu hỏi đã thi")
    transcript = models.TextField(verbose_name="Nội dung trả lời")
    score = models.FloatField(verbose_name="Điểm số")
    feedback = models.TextField(verbose_name="Nhận xét của AI", null=True, blank=True)
    analysis = models.JSONField(verbose_name="Phân tích chi tiết", null=True)
    exam_date = models.DateTimeField(auto_now_add=True, verbose_name="Ngày thi")

    def __str__(self):
        return f"{self.user.username} - {self.subject.subject_code} - {self.score}"