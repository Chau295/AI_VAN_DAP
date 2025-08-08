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
    """
    Hiển thị trang chủ với danh sách các môn thi được lấy động từ database.
    """
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
    """
    Hiển thị trang thi:
    1. Tạo một ExamSession.
    2. Chọn ngẫu nhiên 3 câu hỏi và gán vào Session.
    3. Gửi Session và bộ câu hỏi đến template.
    """
    subject = get_object_or_404(Subject, subject_code=subject_code)
    all_questions = list(subject.questions.all())

    num_questions_to_select = 3
    if len(all_questions) < num_questions_to_select:
        messages.error(request, f"Môn học '{subject.name}' không có đủ {num_questions_to_select} câu hỏi để tạo đề thi.")
        return redirect('dashboard')

    selected_questions = random.sample(all_questions, num_questions_to_select)

    # Tạo một phiên thi mới
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
    """
    API endpoint để nhận và lưu kết quả thi, sau đó trả về ID của kết quả.
    """
    try:
        data = json.loads(request.body)
        session = ExamSession.objects.get(pk=data['session_id'], user=request.user)
        question = Question.objects.get(pk=data['question_id'])

        # Tạo đối tượng ExamResult mới
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
    """
    API endpoint được gọi khi modal kết quả cuối cùng hiện ra,
    đánh dấu là bài thi đã hoàn thành và lưu thời gian.
    """
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
    """
    Hiển thị lịch sử các LẦN THI (ExamSession).
    """
    exam_sessions = ExamSession.objects.filter(user=request.user, is_completed=True).order_by('-created_at')
    context = {
        'sessions': exam_sessions
    }
    return render(request, 'qna/history.html', context)


@login_required
def history_detail_view(request, session_id):
    """
    Hiển thị chi tiết một LẦN THI (ExamSession).
    Bao gồm tất cả câu hỏi trong đề và kết quả (nếu có).
    """
    session = get_object_or_404(ExamSession, pk=session_id, user=request.user)
    all_questions_in_session = session.questions.all()
    results = {result.question_id: result for result in session.results.all()}

    detailed_questions = []
    for question in all_questions_in_session:
        result = results.get(question.id)
        detailed_questions.append({
            'question': question,
            'result': result
        })

    context = {
        'session': session,
        'detailed_questions': detailed_questions
    }
    return render(request, 'qna/history_detail.html', context)


@login_required
def exam_result_detail_view(request, result_id):
    """
    Hiển thị chi tiết một KẾT QUẢ CÂU TRẢ LỜI (ExamResult) cụ thể.
    """
    exam_result = get_object_or_404(ExamResult, pk=result_id, session__user=request.user)
    context = {
        'exam': exam_result
    }
    return render(request, 'qna/exam_result_detail.html', context)


@login_required
def profile_view(request):
    """
    Hiển thị trang thông tin cá nhân của người dùng.
    """
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
    """
    API endpoint để xử lý việc tải lên và cập nhật ảnh đại diện.
    """
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