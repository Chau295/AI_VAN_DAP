# qna/views.py

import random
import json
import logging
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.contrib import messages
from .models import UserProfile, Subject, Question, ExamResult, User
from .forms import RegistrationForm

logger = logging.getLogger(__name__)

def register_view(request):
    if request.method == 'POST':
        form = RegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            username = form.cleaned_data.get('username')
            messages.success(request, f'Tài khoản {username} đã được tạo thành công! Vui lòng đăng nhập.')
            return redirect('login')
        else:
            for field, errors in form.errors.items():
                for error in errors:
                    if field == '__all__':
                        messages.error(request, error)
                    else:
                        field_label = form.fields[field].label
                        messages.error(request, f"Lỗi ở trường '{field_label}': {error}")
    else:
        form = RegistrationForm()

    return render(request, 'registration/register.html', {'form': form})


@login_required
def dashboard_view(request):
    try:
        user_full_name = request.user.userprofile.full_name or request.user.username
    except UserProfile.DoesNotExist:
        user_full_name = request.user.username

    subjects = Subject.objects.all()
    context = {
        'user_full_name': user_full_name,
        'subjects': subjects,
    }
    return render(request, 'qna/dashboard.html', context)


@login_required
def exam_view(request, subject_code):
    subject = get_object_or_404(Subject, subject_code=subject_code)
    question_list = list(subject.questions.all())

    if not question_list:
        messages.error(request, f"Môn học '{subject.name}' hiện chưa có câu hỏi nào.")
        return redirect('dashboard')

    random_question = random.choice(question_list)
    context = {
        'subject': subject,
        'question': random_question,
    }
    return render(request, 'qna/exam.html', context)


@require_POST
@login_required
def save_exam_result(request):
    try:
        data = json.loads(request.body)
        question = Question.objects.get(pk=data['question_id'])

        ExamResult.objects.create(
            user=request.user,
            question=question,
            subject=question.subject,
            transcript=data['transcript'],
            score=data['final_score'],
            feedback=data.get('feedback', ''),
            analysis=data.get('analysis', [])
        )
        return JsonResponse({'status': 'success', 'message': 'Kết quả đã được lưu thành công.'})
    except Exception as e:
        logger.error(f"Lỗi khi lưu kết quả thi cho user {request.user.username}: {e}")
        return JsonResponse({'status': 'error', 'message': str(e)}, status=400)


@login_required
def history_view(request):
    exam_history = ExamResult.objects.filter(user=request.user).select_related('subject').order_by('-exam_date')
    context = {
        'exams': exam_history
    }
    return render(request, 'qna/history.html', context)


@login_required
def history_detail_view(request, exam_id):
    exam_detail = get_object_or_404(ExamResult, pk=exam_id, user=request.user)
    context = {
        'exam': exam_detail
    }
    return render(request, 'qna/history_detail.html', context)


@login_required
def profile_view(request):
    user_profile, created = UserProfile.objects.get_or_create(user=request.user)

    context = {
        'user_profile': user_profile,
        'full_name': user_profile.full_name or request.user.username,
        'student_id': user_profile.student_id or "Chưa cập nhật",
        'class_name': user_profile.class_name or "Chưa cập nhật",
        'email': request.user.email or "Chưa cập nhật",
        'faculty': 'Khoa Hệ thống thông tin',
    }
    return render(request, 'qna/profile.html', context)


@require_POST
@login_required
def update_profile_image(request):
    if 'profile_image' in request.FILES:
        user_profile, created = UserProfile.objects.get_or_create(user=request.user)
        user_profile.profile_image = request.FILES['profile_image']
        user_profile.save()

        return JsonResponse({
            'success': True,
            'message': 'Ảnh đại diện đã được cập nhật thành công.',
            'image_url': user_profile.profile_image.url
        })
    return JsonResponse({'success': False, 'message': 'Không có tệp ảnh nào được gửi lên.'})