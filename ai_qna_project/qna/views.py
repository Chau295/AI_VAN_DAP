# qna/views.py

import random
import json
import logging
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.contrib import messages
from django.utils import timezone
from .models import UserProfile, Subject, Question, ExamResult, User, ExamSession
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
    all_questions = list(subject.questions.all())

    num_questions_to_select = 3
    if len(all_questions) < num_questions_to_select:
        messages.error(request,
                       f"Môn học '{subject.name}' không có đủ {num_questions_to_select} câu hỏi để tạo đề thi.")
        return redirect('dashboard')

    selected_questions = random.sample(all_questions, num_questions_to_select)

    session = ExamSession.objects.create(user=request.user, subject=subject)
    session.questions.set(selected_questions)

    context = {
        'subject': subject,
        'session': session,
        'selected_questions': selected_questions,
    }
    return render(request, 'qna/exam.html', context)


@require_POST
@login_required
def save_exam_result(request):
    try:
        data = json.loads(request.body)
        session = ExamSession.objects.get(pk=data['session_id'], user=request.user)
        question = Question.objects.get(pk=data['question_id'])
        exam_result = ExamResult.objects.create(
            session=session,
            question=question,
            transcript=data['transcript'],
            score=data['final_score'],
            feedback=data.get('feedback', ''),
            analysis=data.get('analysis', [])
        )
        return JsonResponse({
            'status': 'success',
            'message': 'Kết quả đã được lưu thành công.',
            'exam_result_id': exam_result.id
        })
    except Exception as e:
        logger.error(f"Lỗi khi lưu kết quả thi cho user {request.user.username}: {e}")
        return JsonResponse({'status': 'error', 'message': str(e)}, status=400)


@require_POST
@login_required
def complete_exam_session(request):
    try:
        data = json.loads(request.body)
        session = ExamSession.objects.get(pk=data['session_id'], user=request.user)
        if not session.completed_at:
            session.completed_at = timezone.now()
            session.is_completed = True
            session.save()
        return JsonResponse({'status': 'success'})
    except Exception as e:
        logger.error(f"Lỗi khi hoàn thành session: {e}")
        return JsonResponse({'status': 'error', 'message': str(e)}, status=400)


@login_required
def history_view(request):
    exam_sessions = ExamSession.objects.filter(user=request.user, is_completed=True).order_by('-created_at')
    context = {
        'sessions': exam_sessions
    }
    return render(request, 'qna/history.html', context)


@login_required
def history_session_detail_api(request, session_id):
    """
    API trả về dữ liệu chi tiết của một phiên thi dưới dạng JSON.
    """
    session = get_object_or_404(ExamSession, pk=session_id, user=request.user)
    all_questions_in_session = session.questions.all()
    results = {result.question_id: result for result in session.results.all()}

    detailed_questions = []
    for question in all_questions_in_session:
        result = results.get(question.id)
        detailed_questions.append({
            'question_text': question.question_text,
            'result_id': result.id if result else None,
            'score': result.score if result else 0.0,
        })

    # SỬA LỖI Ở ĐÂY: Thay đổi '%H:%i' thành '%H:%M'
    completed_at_str = session.completed_at.strftime('%d/%m/%Y %H:%M') if session.completed_at else "Chưa hoàn thành"

    response_data = {
        'subject_name': session.subject.name,
        'completed_at': completed_at_str,
        'average_score': round(session.calculate_average_score(), 2),
        'detailed_questions': detailed_questions,
        'is_re_evaluation_allowed': session.is_re_evaluation_allowed(),
    }
    return JsonResponse(response_data)


@login_required
def history_result_detail_api(request, result_id):
    """
    API trả về dữ liệu chi tiết của một câu trả lời dưới dạng JSON.
    """
    exam_result = get_object_or_404(ExamResult, pk=result_id, session__user=request.user)

    response_data = {
        'question_text': exam_result.question.question_text,
        'transcript': exam_result.transcript,
        'analysis': exam_result.analysis,
        'feedback': exam_result.feedback,
        'score': round(exam_result.score, 2),
    }
    return JsonResponse(response_data)


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