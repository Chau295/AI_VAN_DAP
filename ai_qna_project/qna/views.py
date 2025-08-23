# -*- coding: utf-8 -*-
"""
Module này chứa tất cả các view và logic xử lý request cho ứng dụng Q&A.

Bao gồm các chức năng chính:
- Đăng ký, đăng nhập, quản lý hồ sơ người dùng.
- Hiển thị dashboard, lịch sử thi.
- Luồng thực hiện một phiên thi (câu hỏi chính và phụ).
- Các API endpoints để lưu kết quả, lấy câu hỏi, và cập nhật avatar.
"""

from __future__ import annotations

import json
import random
from base64 import b64encode
from typing import Optional, List, Dict, Any

from django import forms
from django.contrib import messages
from django.contrib.auth import get_user_model, login
from django.contrib.auth.decorators import login_required
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import PermissionDenied, ValidationError
from django.db.models import Sum
from django.http import (
    JsonResponse,
    HttpRequest,
    HttpResponse,
    HttpResponseBadRequest,
)
from django.shortcuts import get_object_or_404, render, redirect
from django.templatetags.static import static
from django.utils import timezone
from django.utils.safestring import mark_safe
from django.views.decorators.http import require_POST, require_GET

from .models import (
    Subject,
    Question,
    ExamSession,
    ExamResult,
    SupplementaryResult,
    UserProfile,
)

User = get_user_model()

# ===========================
# CÁC QUY TẮC VÀ HẰNG SỐ
# ===========================
SUPP_MAX_PER_QUESTION = 1.0  # Điểm tối đa cho mỗi câu hỏi phụ
SUPP_MAX_COUNT = 2  # Số lượng câu hỏi phụ tối đa được tính điểm
FINAL_CAP = 7.0  # Điểm cuối cùng tối đa nếu có trả lời câu hỏi phụ


# ===========================
# CÁC HÀM HỖ TRỢ (HELPERS)
# ===========================

def _json_body(request: HttpRequest) -> Dict[str, Any]:
    """Tải nội dung JSON từ body của request một cách an toàn."""
    try:
        if request.body:
            return json.loads(request.body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        pass
    return {}


def _ensure_owner(session: ExamSession, user: User) -> None:
    """Kiểm tra người dùng hiện tại có phải là chủ sở hữu của phiên thi không."""
    if session.user_id != getattr(user, "id", None):
        raise PermissionDenied("Bạn không có quyền truy cập phiên thi này.")


def _compute_scores(session: ExamSession) -> tuple[float, float, float]:
    """
    Tính toán và trả về bộ 3 điểm của một phiên thi.
    Xử lý phòng thủ với dữ liệu điểm không nhất quán từ DB.
    """
    # 1. Tính điểm trung bình phần chính (Logic không đổi)
    total_main_questions = session.questions.count()
    if total_main_questions == 0:
        total_main_questions = 3

    main_results = ExamResult.objects.filter(session=session)
    sum_of_scores = main_results.aggregate(total=Sum('score'))['total'] or 0.0
    main_avg = sum_of_scores / total_main_questions

    # 2. Tính tổng điểm câu hỏi phụ (lấy 2 câu cao nhất sau khi khử lặp và làm sạch dữ liệu)
    supp_results_qs = SupplementaryResult.objects.filter(session=session)
    best_scores_by_text = {}
    for result in supp_results_qs:
        key = (result.question_text or "").strip()
        if not key:
            continue

        score = float(result.score or 0.0)

        # ▼▼▼ LOGIC LÀM SẠCH DỮ LIỆU ĐƯỢC THÊM VÀO ▼▼▼
        # Nếu điểm > 1.0, giả định đó là thang 10 và quy đổi về thang 1
        if score > SUPP_MAX_PER_QUESTION:
            score /= 10.0
        # ▲▲▲ KẾT THÚC LOGIC LÀM SẠCH ▲▲▲

        current_score = max(0.0, min(score, SUPP_MAX_PER_QUESTION))

        if key not in best_scores_by_text or current_score > best_scores_by_text[key]:
            best_scores_by_text[key] = current_score

    unique_supp_scores = list(best_scores_by_text.values())
    unique_supp_scores.sort(reverse=True)

    supp_sum = sum(unique_supp_scores[:SUPP_MAX_COUNT])

    # 3. Tính điểm cuối cùng (Logic không đổi)
    if supp_sum > 0:
        final_total = min(FINAL_CAP, main_avg + supp_sum)
    else:
        final_total = min(10.0, main_avg)

    return main_avg, supp_sum, final_total


# qna/views.py

def _dedupe_supp_for_display(qs: SupplementaryResult) -> List[SupplementaryResult]:
    """
    Khử lặp các câu hỏi phụ và làm sạch điểm để hiển thị.
    """
    best_by_text = {}
    for result in qs:
        key = (result.question_text or "").strip()
        if not key:
            continue

        # ▼▼▼ LOGIC LÀM SẠCH DỮ LIỆU ĐƯỢC THÊM VÀO ▼▼▼
        # Tạo một bản sao để không thay đổi đối tượng gốc trong queryset
        cleaned_result = result
        score = float(cleaned_result.score or 0.0)

        # Nếu điểm > 1.0, giả định đó là thang 10 và quy đổi về thang 1
        if score > SUPP_MAX_PER_QUESTION:
            cleaned_result.score = score / 10.0
        # ▲▲▲ KẾT THÚC LOGIC LÀM SẠCH ▲▲▲

        current_best = best_by_text.get(key)
        if current_best is None or float(cleaned_result.score or 0) > float(current_best.score or 0):
            best_by_text[key] = cleaned_result

    items = list(best_by_text.values())
    items.sort(key=lambda x: float(x.score or 0), reverse=True)
    return items[:SUPP_MAX_COUNT]

# ===========================
# FORM ĐĂNG KÝ
# ===========================

class RegistrationForm(forms.Form):
    """Form xử lý việc đăng ký tài khoản mới."""
    full_name = forms.CharField(
        label=mark_safe('Họ và tên <span class="text-red-500">*</span>'),
        max_length=150,
        widget=forms.TextInput(attrs={"placeholder": "VD: Nguyễn Văn A", "autocomplete": "name"})
    )
    username = forms.CharField(
        label=mark_safe('Tên đăng nhập (Mã SV) <span class="text-red-500">*</span>'),
        max_length=150,
        widget=forms.TextInput(attrs={"placeholder": "Mã SV hoặc tên đăng nhập", "autocomplete": "username"}),
        help_text="Phải là duy nhất."
    )
    class_name = forms.CharField(
        label=mark_safe('Lớp <span class="text-red-500">*</span>'),
        max_length=100,
        widget=forms.TextInput(attrs={"placeholder": "VD: K25CNTT", "autocomplete": "organization"})
    )
    email = forms.EmailField(
        label='Email',
        required=False,
        widget=forms.EmailInput(attrs={"placeholder": "ten@sv.duytan.edu.vn", "autocomplete": "email"}),
        help_text="(Không bắt buộc)"
    )
    faculty = forms.CharField(
        label='Khoa',
        required=False,
        max_length=150,
        widget=forms.TextInput(attrs={"placeholder": "VD: Công nghệ thông tin"}),
        help_text="(Không bắt buộc)"
    )
    password = forms.CharField(
        label=mark_safe('Mật khẩu <span class="text-red-500">*</span>'),
        strip=False,
        widget=forms.PasswordInput(attrs={"placeholder": "••••••••", "autocomplete": "new-password"}),
        help_text="Tối thiểu 8 ký tự."
    )
    password2 = forms.CharField(
        label=mark_safe('Nhập lại mật khẩu <span class="text-red-500">*</span>'),
        strip=False,
        widget=forms.PasswordInput(attrs={"placeholder": "••••••••", "autocomplete": "new-password"}),
        help_text="Phải trùng với mật khẩu."
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields.values():
            field.widget.attrs.update({
                "class": "appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:ring-2 focus:ring-blue-400"
            })

    def clean_username(self):
        """Kiểm tra tên đăng nhập không được trống và phải là duy nhất."""
        username = self.cleaned_data.get("username", "").strip()
        if not username:
            raise ValidationError("Tên đăng nhập là bắt buộc.")
        if User.objects.filter(username=username).exists():
            raise ValidationError("Tên đăng nhập này đã tồn tại.")
        return username

    def clean(self):
        """Kiểm tra mật khẩu và các ràng buộc toàn form."""
        cleaned_data = super().clean()
        password = cleaned_data.get("password")
        password2 = cleaned_data.get("password2")

        if password and password2 and password != password2:
            self.add_error("password2", "Mật khẩu nhập lại không khớp.")

        if password:
            try:
                validate_password(password, user=User(username=cleaned_data.get("username")))
            except ValidationError as e:
                self.add_error("password", e)

        return cleaned_data


# ===========================
# CÁC VIEW HIỂN THỊ TRANG (PAGES)
# ===========================

def register_view(request: HttpRequest) -> HttpResponse:
    """Xử lý trang đăng ký tài khoản."""
    if request.user.is_authenticated:
        return redirect("qna:dashboard")

    if request.method == "POST":
        form = RegistrationForm(request.POST)
        if form.is_valid():
            data = form.cleaned_data
            user = User.objects.create_user(
                username=data["username"],
                password=data["password"],
                email=data.get("email", ""),
                first_name=data["full_name"]
            )

            profile, _ = UserProfile.objects.get_or_create(user=user)
            profile.full_name = data["full_name"]
            profile.class_name = data["class_name"]
            profile.student_id = data["username"]
            profile.faculty = data.get("faculty", "")
            profile.save()

            login(request, user)
            messages.success(request, "Đăng ký thành công. Chào mừng bạn!")
            return redirect("qna:dashboard")
    else:
        form = RegistrationForm()

    return render(request, "registration/register.html", {"form": form})


@login_required
def dashboard_view(request: HttpRequest) -> HttpResponse:
    """Hiển thị trang dashboard chính sau khi đăng nhập."""
    subjects = Subject.objects.all().order_by("name")
    recent_sessions = (
        ExamSession.objects.filter(user=request.user)
        .select_related("subject")
        .order_by("-created_at")[:5]
    )
    profile, _ = UserProfile.objects.get_or_create(user=request.user)

    return render(request, "qna/dashboard.html", {
        "subjects": subjects,
        "recent_sessions": recent_sessions,
        "full_name": profile.full_name or request.user.first_name,
    })


# qna/views.py

@login_required
def history_view(request: HttpRequest) -> HttpResponse:
    """Hiển thị trang lịch sử các phiên thi của người dùng."""
    sessions = (
        ExamSession.objects.filter(user=request.user)
        .select_related("subject")
        .order_by("-created_at")
    )
    for s in sessions:
        main_avg, supp_sum, final_total = _compute_scores(s)
        s.main_avg = main_avg
        s.supp_sum = supp_sum
        # ▼▼▼ THAY ĐỔI Ở ĐÂY ▼▼▼
        # Đổi tên thuộc tính để tránh xung đột và làm rõ ý nghĩa.
        s.calculated_final_score = final_total
        # ▲▲▲ KẾT THÚC THAY ĐỔI ▲▲▲

    return render(request, "qna/history.html", {"sessions": sessions})

@login_required
def history_detail_view(request: HttpRequest, session_id: int) -> HttpResponse:
    """Hiển thị chi tiết kết quả của một phiên thi."""
    session = get_object_or_404(
        ExamSession.objects.select_related("subject", "user"),
        pk=session_id,
    )
    _ensure_owner(session, request.user)

    main_results = (
        ExamResult.objects.filter(session=session)
        .select_related("question")
        .order_by("question_id")
    )
    supp_results_qs = SupplementaryResult.objects.filter(session=session)
    supp_results_display = _dedupe_supp_for_display(supp_results_qs)

    main_avg, supp_sum, final_total = _compute_scores(session)

    return render(request, "qna/history_detail.html", {
        "session": session,
        "results": main_results,
        "supp_results": supp_results_display,
        "main_avg": main_avg,
        "supp_sum": supp_sum,
        "final_total": final_total,
    })


@login_required
def exam_view(request: HttpRequest, subject_code: str) -> HttpResponse:
    """Bắt đầu một phiên thi mới cho một môn học."""
    subject = get_object_or_404(Subject, subject_code=subject_code)

    main_questions = list(
        Question.objects.filter(subject=subject, is_supplementary=False).order_by("?")[:3]
    )

    if not main_questions:
        messages.error(request, f"Môn {subject.name} chưa có câu hỏi. Vui lòng liên hệ quản trị viên.")
        return redirect("qna:dashboard")

    session = ExamSession.objects.create(user=request.user, subject=subject)
    session.questions.set(main_questions)

    remaining_main_qs = Question.objects.filter(
        subject=subject, is_supplementary=False
    ).exclude(id__in=[q.id for q in main_questions])

    barem = [{"id": q.id, "question": q.question_text} for q in remaining_main_qs]

    return render(request, "qna/exam.html", {
        "subject": subject,
        "selected_questions": main_questions,
        "session": session,
        "barem_json": json.dumps(barem, ensure_ascii=False),
    })


# ===========================
# HỒ SƠ VÀ AVATAR
# ===========================

def _get_avatar_data_url(profile: UserProfile) -> str:
    """Tạo chuỗi data URL cho avatar từ DB blob hoặc trả về ảnh mặc định."""
    if profile.profile_image_blob:
        try:
            mime = profile.profile_image_mime or "image/jpeg"
            encoded_blob = b64encode(profile.profile_image_blob).decode("ascii")
            return f"data:{mime};base64,{encoded_blob}"
        except Exception:
            pass
    return static("images/default_avatar.png")


@login_required
def profile_view(request: HttpRequest) -> HttpResponse:
    """Hiển thị trang hồ sơ người dùng (chỉ đọc)."""
    profile, _ = UserProfile.objects.get_or_create(user=request.user)

    context = {
        "username": request.user.username,
        "email": request.user.email,
        "full_name": profile.full_name or request.user.first_name,
        "class_name": profile.class_name,
        "student_id": profile.student_id,
        "date_joined": request.user.date_joined,
        "last_login": request.user.last_login,
        "avatar_url": _get_avatar_data_url(profile),
    }
    return render(request, "qna/profile.html", context)


@login_required
@require_POST
def update_profile_image(request: HttpRequest) -> JsonResponse:
    """API để tải lên và cập nhật ảnh đại diện mới, lưu dưới dạng blob."""
    file_obj = request.FILES.get('profile_image')
    if not file_obj:
        return JsonResponse({"success": False, "error": "Không tìm thấy file ảnh."}, status=400)

    if file_obj.size > 5 * 1024 * 1024:  # Giới hạn 5MB
        return JsonResponse({"success": False, "error": "Kích thước ảnh không được vượt quá 5MB."}, status=400)

    content = file_obj.read()
    mime = file_obj.content_type

    profile, _ = UserProfile.objects.get_or_create(user=request.user)

    profile.profile_image_blob = content
    profile.profile_image_mime = mime
    profile.save()

    data_url = f"data:{mime};base64,{b64encode(content).decode('ascii')}"
    return JsonResponse({"success": True, "image_data_url": data_url})


# ===========================
# CÁC API CHO LUỒNG THI
# ===========================

@login_required
@require_POST
def save_exam_result(request: HttpRequest) -> JsonResponse:
    """API để lưu kết quả của một câu hỏi chính."""
    data = _json_body(request)
    session_id = data.get("session_id")
    question_id = data.get("question_id")
    score = data.get("score")

    if not all([session_id, question_id, score is not None]):
        return HttpResponseBadRequest("Thiếu các tham số bắt buộc (session_id, question_id, score).")

    session = get_object_or_404(ExamSession, pk=session_id)
    _ensure_owner(session, request.user)
    question = get_object_or_404(Question, pk=question_id, is_supplementary=False)

    result, created = ExamResult.objects.update_or_create(
        session=session,
        question=question,
        defaults={
            "transcript": data.get("transcript", ""),
            "score": float(score),
            "feedback": data.get("feedback"),
            "analysis": data.get("analysis"),
            "answered_at": timezone.now(),
        }
    )
    return JsonResponse({"status": "ok", "created": created, "result_id": result.id})


@login_required
@require_POST
def get_supplementary_for_session(request: HttpRequest, session_id: int) -> JsonResponse:
    """API để lấy ngẫu nhiên 2 câu hỏi phụ cho một phiên thi."""
    session = get_object_or_404(ExamSession.objects.select_related("subject"), pk=session_id)
    _ensure_owner(session, request.user)

    supp_pool = list(Question.objects.filter(subject=session.subject, is_supplementary=True))
    random.shuffle(supp_pool)

    picked_questions = supp_pool[:SUPP_MAX_COUNT]
    items = [{"id": q.id, "question": q.question_text} for q in picked_questions]

    return JsonResponse({"status": "ok", "items": items})


@login_required
@require_POST
def save_supplementary_result(request: HttpRequest) -> JsonResponse:
    """API để lưu kết quả của một câu hỏi phụ."""
    data = _json_body(request)
    session_id = data.get("session_id")
    question_text = (data.get("question_text") or "").strip()
    raw_score = data.get("score")

    if not all([session_id, question_text, raw_score is not None]):
        return HttpResponseBadRequest("Thiếu tham số (session_id, question_text, score).")

    session = get_object_or_404(ExamSession, pk=session_id)
    _ensure_owner(session, request.user)

    if SupplementaryResult.objects.filter(session=session).count() >= SUPP_MAX_COUNT:
        return JsonResponse(
            {"status": "error", "message": f"Đã đạt số lượng câu hỏi phụ tối đa ({SUPP_MAX_COUNT})."},
            status=400,
        )

    try:
        score_val = float(raw_score)
        max_score_val = float(data.get("max_score", 10.0))
        if max_score_val <= 0: max_score_val = 10.0
    except (TypeError, ValueError):
        return HttpResponseBadRequest("Giá trị điểm không hợp lệ.")

    normalized_score = (score_val / max_score_val) * SUPP_MAX_PER_QUESTION
    final_score = max(0.0, min(normalized_score, SUPP_MAX_PER_QUESTION))

    sr = SupplementaryResult.objects.create(
        session=session,
        question_text=question_text,
        transcript=data.get("transcript", ""),
        score=final_score,
        max_score=SUPP_MAX_PER_QUESTION,
        feedback=data.get("feedback"),
        analysis=data.get("analysis"),
    )

    main_avg, supp_sum, final_total = _compute_scores(session)
    return JsonResponse({
        "status": "ok",
        "supplementary_result_id": sr.id,
        "main_avg": main_avg,
        "supp_sum": supp_sum,
        "final_total": final_total,
    })


@login_required
@require_POST
def finalize_session_view(request: HttpRequest, session_id: int) -> JsonResponse:
    """API để hoàn thành một phiên thi."""
    session = get_object_or_404(ExamSession, pk=session_id)
    _ensure_owner(session, request.user)

    # Dùng hàm tính điểm đã được cập nhật
    main_avg, _, total_score = _compute_scores(session)

    session.is_completed = True
    session.completed_at = timezone.now()
    session.final_score = total_score
    session.save(update_fields=["is_completed", "completed_at", "final_score"])

    return JsonResponse({"status": "success", "final_score": session.final_score})