# ai_qna_project/qna/models.py
from django.db import models
from django.utils import timezone
from django.contrib.auth.models import User
from django.db.models import Avg
from datetime import timedelta


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

    def __str__(self):
        return f"{self.subject.subject_code} - {self.question_text[:50]}..."


class ExamSession(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='exam_sessions')
    subject = models.ForeignKey(Subject, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="Ngày thi")
    completed_at = models.DateTimeField(null=True, blank=True, verbose_name="Thời gian hoàn thành")
    questions = models.ManyToManyField(Question, related_name='exam_sessions')
    is_completed = models.BooleanField(default=False)

    def calculate_average_score(self):
        total_questions = self.questions.count()
        if not total_questions:
            return 0.0

        answered_results = self.results.all()
        total_score = sum(result.score for result in answered_results)

        # Điểm trung bình được tính trên tổng số câu hỏi trong đề (bao gồm cả câu 0 điểm)
        return total_score / total_questions

    def is_re_evaluation_allowed(self):
        if not self.completed_at:
            return False
        # Giới hạn phúc khảo trong 10 phút sau khi hoàn thành
        return timezone.now() <= self.completed_at + timedelta(minutes=10)

    def get_re_evaluation_remaining_time(self):
        """
        Trả về thời gian phúc khảo còn lại (tính bằng giây).
        Nếu hết hạn, trả về 0.
        """
        if not self.completed_at:
            return 0

        deadline = self.completed_at + timedelta(minutes=10)
        now = timezone.now()

        if now >= deadline:
            return 0

        remaining = deadline - now
        return int(remaining.total_seconds())

    def __str__(self):
        return f"Bài thi môn {self.subject.name} của {self.user.username} ngày {self.created_at.strftime('%d/%m/%Y')}"


class ExamResult(models.Model):
    session = models.ForeignKey(ExamSession, on_delete=models.CASCADE, related_name='results')
    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    transcript = models.TextField(verbose_name="Nội dung trả lời")
    score = models.FloatField(verbose_name="Điểm số")
    feedback = models.TextField(verbose_name="Nhận xét của AI", null=True, blank=True)
    analysis = models.JSONField(verbose_name="Phân tích chi tiết", null=True)
    answered_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Kết quả câu hỏi {self.question.id} của {self.session.user.username}"