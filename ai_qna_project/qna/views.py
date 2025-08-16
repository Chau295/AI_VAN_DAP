# qna/views.py
from __future__ import annotations

import json
import random
from typing import Tuple, Optional

from django import forms
from django.contrib import messages
from django.contrib.auth import get_user_model, login
from django.contrib.auth.decorators import login_required
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import PermissionDenied, ValidationError
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
    UserProfile,  # dùng cho hồ sơ & avatar
)

User = get_user_model()


# ===========================
# Helpers
# ===========================

def _json_body(request: HttpRequest) -> dict:
    """Safe JSON loader for request.body; returns {} if empty/invalid."""
    try:
        if request.body:
            return json.loads(request.body.decode("utf-8"))
    except Exception:
        pass
    return {}

def _ensure_owner(session: ExamSession, user) -> None:
    if session.user_id != getattr(user, "id", None):
        raise PermissionDenied("Bạn không có quyền với phiên thi này.")

def _compute_scores(session: ExamSession) -> Tuple[Optional[float], float, float]:
    """
    Trả về bộ 3:
      - main_avg: điểm trung bình phần chính (None nếu chưa có câu nào)
      - supp_sum: tổng điểm cộng từ câu phụ
      - final_total: tổng điểm sau khi cộng câu phụ, giới hạn tối đa 10.0
    """
    mains = list(ExamResult.objects.filter(session=session).only("score"))
    supp_sum = sum(
        (sr.score or 0.0)
        for sr in SupplementaryResult.objects.filter(session=session).only("score")
    )

    if not mains:
        return None, supp_sum, 0.0

    main_avg = sum((er.score or 0.0) for er in mains) / len(mains)
    final_total = min(10.0, main_avg + supp_sum)
    return main_avg, supp_sum, final_total


# ===========================
# Registration (Đăng ký)
# ===========================

class RegistrationForm(forms.Form):
    # Các trường bắt buộc có dấu * đỏ (hiển thị trong label)
    full_name = forms.CharField(
        label=mark_safe('Họ và tên <span class="text-red-500">*</span>'),
        required=True,
        max_length=150,
        widget=forms.TextInput(attrs={
            "class": "appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:ring-2 focus:ring-blue-400",
            "placeholder": "VD: Nguyễn Văn A",
            "autocomplete": "name",
        }),
        help_text="",
    )
    username = forms.CharField(
        label=mark_safe('Tên đăng nhập (Mã SV) <span class="text-red-500">*</span>'),
        required=True,
        max_length=150,
        widget=forms.TextInput(attrs={
            "class": "appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:ring-2 focus:ring-blue-400",
            "placeholder": "Mã SV hoặc tên đăng nhập",
            "autocomplete": "username",
        }),
        help_text="Phải là duy nhất.",
    )
    class_name = forms.CharField(
        label=mark_safe('Lớp <span class="text-red-500">*</span>'),
        required=True,
        max_length=100,
        widget=forms.TextInput(attrs={
            "class": "appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:ring-2 focus:ring-blue-400",
            "placeholder": "VD: K25CNTT",
            "autocomplete": "organization",
        }),
        help_text="",
    )
    # Không bắt buộc → KHÔNG gắn *
    email = forms.EmailField(
        label='Email',
        required=False,
        widget=forms.EmailInput(attrs={
            "class": "appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:ring-2 focus:ring-blue-400",
            "placeholder": "ten@sv.duytan.edu.vn",
            "autocomplete": "email",
        }),
        help_text="(Không bắt buộc)",
    )
    faculty = forms.CharField(
        label='Khoa',
        required=False,
        max_length=150,
        widget=forms.TextInput(attrs={
            "class": "appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:ring-2 focus:ring-blue-400",
            "placeholder": "VD: Công nghệ thông tin",
        }),
        help_text="(Không bắt buộc)",
    )
    password = forms.CharField(
        label=mark_safe('Mật khẩu <span class="text-red-500">*</span>'),
        required=True,
        strip=False,
        widget=forms.PasswordInput(attrs={
            "class": "appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:ring-2 focus:ring-blue-400",
            "placeholder": "••••••••",
            "autocomplete": "new-password",
        }),
        help_text="Tối thiểu 8 ký tự, nên có chữ hoa, chữ thường, số và ký tự đặc biệt.",
    )
    password2 = forms.CharField(
        label=mark_safe('Nhập lại mật khẩu <span class="text-red-500">*</span>'),
        required=True,
        strip=False,
        widget=forms.PasswordInput(attrs={
            "class": "appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:ring-2 focus:ring-blue-400",
            "placeholder": "••••••••",
            "autocomplete": "new-password",
        }),
        help_text="Phải trùng với mật khẩu.",
    )

    @staticmethod
    def _nonblank(value: str) -> str:
        if value is None:
            raise ValidationError("Trường này là bắt buộc.")
        v = value.strip()
        if not v:
            raise ValidationError("Trường này là bắt buộc.")
        return v

    def clean_full_name(self):
        return self._nonblank(self.cleaned_data.get("full_name"))

    def clean_username(self):
        username = self._nonblank(self.cleaned_data.get("username"))
        if User.objects.filter(username=username).exists():
            raise ValidationError("Tên đăng nhập đã tồn tại.")
        return username

    def clean_class_name(self):
        return self._nonblank(self.cleaned_data.get("class_name"))

    def clean(self):
        cleaned = super().clean()

        pwd = (cleaned.get("password") or "").strip()
        pwd2 = (cleaned.get("password2") or "").strip()

        if not pwd:
            self.add_error("password", "Trường này là bắt buộc.")
        if not pwd2:
            self.add_error("password2", "Trường này là bắt buộc.")
        if pwd and pwd2 and pwd != pwd2:
            self.add_error("password2", "Mật khẩu nhập lại không khớp.")

        username = cleaned.get("username") or ""
        # kiểm tra độ mạnh mật khẩu theo rule mặc định của Django
        if pwd:
            temp_user = User(username=username)
            try:
                validate_password(pwd, user=temp_user)
            except ValidationError as e:
                self.add_error("password", e)

        return cleaned


# -------- Profile update form (Email & Khoa) --------
class ProfileUpdateForm(forms.Form):
    email = forms.EmailField(
        label="Email",
        required=False,
        widget=forms.EmailInput(attrs={
            "class": "appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:ring-2 focus:ring-blue-400",
            "placeholder": "ten@sv.duytan.edu.vn",
            "autocomplete": "email",
        }),
        help_text="",
    )
    faculty = forms.CharField(
        label="Khoa",
        required=False,
        max_length=150,
        widget=forms.TextInput(attrs={
            "class": "appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:ring-2 focus:ring-blue-400",
            "placeholder": "VD: Công nghệ thông tin",
        }),
        help_text="",
    )


def _try_set_profile_fields(
    profile: UserProfile,
    *,
    full_name: Optional[str] = None,
    class_name: Optional[str] = None,
    faculty: Optional[str] = None,
    student_id: Optional[str] = None,
) -> None:
    """
    Gán an toàn các field nếu tồn tại trong UserProfile (tránh vỡ code nếu schema khác).
    """
    changed_fields = []

    if full_name is not None and hasattr(profile, "full_name"):
        try:
            profile.full_name = full_name
            changed_fields.append("full_name")
        except Exception:
            pass

    if class_name is not None and hasattr(profile, "class_name"):
        try:
            profile.class_name = class_name
            changed_fields.append("class_name")
        except Exception:
            pass

    if faculty is not None and hasattr(profile, "faculty"):
        try:
            profile.faculty = faculty
            changed_fields.append("faculty")
        except Exception:
            pass

    if student_id is not None and hasattr(profile, "student_id"):
        try:
            profile.student_id = student_id
            changed_fields.append("student_id")
        except Exception:
            pass

    if changed_fields:
        try:
            profile.save(update_fields=changed_fields)
        except Exception:
            profile.save()


def register_view(request: HttpRequest) -> HttpResponse:
    """
    View đăng ký khớp với template registration/register.html:
    - Nhãn có dấu * màu đỏ cho trường bắt buộc.
    - Kiểm tra rỗng, trùng username, độ mạnh mật khẩu.
    - Tạo User + UserProfile (nếu có), gán first_name = full_name, set email nếu nhập.
    - Tự đăng nhập sau khi đăng ký, chuyển tới dashboard.
    """
    if request.user.is_authenticated:
        return redirect("dashboard")

    if request.method == "POST":
        form = RegistrationForm(request.POST)
        if form.is_valid():
            full_name = form.cleaned_data["full_name"].strip()
            username = form.cleaned_data["username"].strip()
            class_name = form.cleaned_data["class_name"].strip()
            email = (form.cleaned_data.get("email") or "").strip()
            faculty = (form.cleaned_data.get("faculty") or "").strip()
            password = form.cleaned_data["password"]

            # Tạo user
            user = User.objects.create_user(username=username, password=password, email=email or None)
            # Lưu full_name vào first_name để hiển thị nhanh
            try:
                user.first_name = full_name
                if email:
                    user.email = email
                user.save(update_fields=["first_name", "email"] if email else ["first_name"])
            except Exception:
                pass

            # Tạo / cập nhật profile
            profile, _ = UserProfile.objects.get_or_create(user=user)
            _try_set_profile_fields(
                profile,
                full_name=full_name,
                class_name=class_name,
                faculty=faculty or None,
                student_id=username,  # alias MSSV nếu model có field student_id
            )

            # Đăng nhập
            login(request, user)
            messages.success(request, "Đăng ký thành công. Chào mừng bạn!")
            return redirect("dashboard")
    else:
        form = RegistrationForm()

    return render(request, "registration/register.html", {"form": form})


# ===========================
# Profile helpers
# ===========================

def _collect_profile_data(user) -> dict:
    """
    Thu thập dữ liệu hồ sơ để hiển thị:
    - Họ và tên: profile.full_name hoặc user.first_name
    - Lớp: profile.class_name
    - Mã số sinh viên: username (hoặc profile.student_id nếu có)
    - Khoa: profile.faculty (nếu có)
    - Email: user.email
    - Avatar: profile.profile_image hoặc ảnh mặc định
    """
    profile, _ = UserProfile.objects.get_or_create(user=user)

    full_name = getattr(profile, "full_name", None) or (user.first_name or "")
    class_name = getattr(profile, "class_name", None) or ""

    # avatar
    avatar_url = ""
    try:
        if getattr(profile, "profile_image", None) and hasattr(profile.profile_image, "url"):
            avatar_url = profile.profile_image.url
    except Exception:
        avatar_url = ""
    if not avatar_url:
        avatar_url = static("images/default_avatar.png")

    faculty = getattr(profile, "faculty", "") or ""
    student_id = getattr(profile, "student_id", "") or user.username

    return {
        "username": user.username,
        "email": getattr(user, "email", "") or "",
        "full_name": full_name,
        "class_name": class_name,
        "date_joined": getattr(user, "date_joined", None),
        "last_login": getattr(user, "last_login", None),
        "avatar_url": avatar_url,
        "faculty": faculty,
        "student_id": student_id,
        "profile": profile,
    }


# ===========================
# Pages
# ===========================

@login_required
def dashboard_view(request: HttpRequest) -> HttpResponse:
    """Trang chính: liệt kê môn học & lịch sử gần đây."""
    subjects = Subject.objects.all().order_by("name")
    recent_sessions = (
        ExamSession.objects.filter(user=request.user)
        .select_related("subject")
        .order_by("-created_at")[:10]
    )
    return render(request, "qna/dashboard.html", {
        "subjects": subjects,
        "recent_sessions": recent_sessions,
    })

@login_required
def profile_view(request: HttpRequest) -> HttpResponse:
    """
    Trang hồ sơ: hiển thị Họ và tên, Lớp, MSSV, Khoa, Email,
    đồng thời có FORM để cập nhật Email & Khoa.
    """
    ctx = _collect_profile_data(request.user)

    if request.method == "POST":
        form = ProfileUpdateForm(request.POST)
        if form.is_valid():
            email = (form.cleaned_data.get("email") or "").strip()
            faculty = (form.cleaned_data.get("faculty") or "").strip()

            # cập nhật email user
            try:
                request.user.email = email
                request.user.save(update_fields=["email"])
            except Exception:
                pass

            # cập nhật faculty trong profile nếu có field
            profile = ctx["profile"]
            if hasattr(profile, "faculty"):
                try:
                    profile.faculty = faculty
                    profile.save(update_fields=["faculty"])
                except Exception:
                    pass

            messages.success(request, "Cập nhật hồ sơ thành công.")
            return redirect("profile")
    else:
        form = ProfileUpdateForm(initial={
            "email": ctx.get("email", ""),
            "faculty": ctx.get("faculty", ""),
        })

    ctx["form"] = form
    return render(request, "qna/profile.html", ctx)


@login_required
def history_view(request: HttpRequest) -> HttpResponse:
    sessions = (
        ExamSession.objects.filter(user=request.user)
        .select_related("subject")
        .order_by("-created_at")
    )
    # Gắn thuộc tính động cho template
    for s in sessions:
        m, supp, t = _compute_scores(s)
        s.main_avg = m or 0.0
        s.supp_sum = supp or 0.0
        s.final_total = t or 0.0
    return render(request, "qna/history.html", {"sessions": sessions})

@login_required
def history_detail_view(request: HttpRequest, session_id: int) -> HttpResponse:
    session = get_object_or_404(
        ExamSession.objects.select_related("subject", "user"),
        pk=session_id,
    )
    _ensure_owner(session, request.user)

    results = (
        ExamResult.objects.filter(session=session)
        .select_related("question")
        .order_by("question_id")
    )
    supp_results = SupplementaryResult.objects.filter(session=session).order_by("created_at")

    main_avg, supp_sum, final_total = _compute_scores(session)

    return render(request, "qna/history_detail.html", {
        "session": session,
        "results": results,
        "supp_results": supp_results,
        "main_avg": main_avg,        # có thể là None; template dùng |default:0 nếu cần
        "supp_sum": supp_sum,
        "final_total": final_total,
    })


# ===========================
# Exam flow (luồng mới)
# ===========================

@login_required
def exam_view(request: HttpRequest, subject_code: str) -> HttpResponse:
    """
    LUỒNG MỚI:
    - Chỉ chọn & hiển thị 3 câu hỏi CHÍNH.
    - Đồng thời gửi xuống BAREM (JSON) chứa pool câu hỏi chính còn lại để client random 2 câu phụ.
    """
    subject = get_object_or_404(Subject, subject_code=subject_code)

    main_pool_qs = Question.objects.filter(subject=subject, is_supplementary=False)

    # Luôn 3 câu chính
    selected_questions = list(main_pool_qs.order_by("?")[:3])

    # Tạo session và gắn câu hỏi
    session = ExamSession.objects.create(user=request.user, subject=subject)
    if selected_questions:
        session.questions.set(selected_questions)

    # BAREM cho câu hỏi phụ: lấy từ "câu hỏi chính còn lại" (loại trừ 3 câu đã chọn)
    remaining_main = main_pool_qs.exclude(
        id__in=[q.id for q in selected_questions]
    ).values("id", "question_text")
    barem = [{"id": r["id"], "question": r["question_text"]} for r in remaining_main]

    return render(request, "qna/exam.html", {
        "subject": subject,
        "selected_questions": selected_questions,
        "session": session,
        "barem_json": json.dumps(barem, ensure_ascii=False),  # client dùng window.__BAREM__
    })


# ===========================
# APIs used by exam flow
# ===========================

@login_required
@require_POST
def save_exam_result(request: HttpRequest) -> JsonResponse:
    """
    Lưu kết quả một câu hỏi CHÍNH.
    Body JSON:
      { session_id, question_id, transcript, score, feedback?, analysis? (list|dict) }
    """
    data = _json_body(request)
    session_id = data.get("session_id")
    question_id = data.get("question_id")
    transcript = data.get("transcript") or ""
    score = data.get("score")
    feedback = data.get("feedback")
    analysis = data.get("analysis")

    if session_id is None or question_id is None or score is None:
        return HttpResponseBadRequest("Thiếu tham số.")

    session = get_object_or_404(ExamSession, pk=session_id)
    _ensure_owner(session, request.user)
    question = get_object_or_404(Question, pk=question_id, is_supplementary=False)

    er, created = ExamResult.objects.update_or_create(
        session=session, question=question,
        defaults={
            "transcript": transcript,
            "score": float(score),
            "feedback": feedback,
            "analysis": analysis,
            "answered_at": timezone.now(),
        }
    )
    return JsonResponse({
        "status": "ok",
        "created": created,
        "result_id": er.id,
    })


# (Không còn dùng nếu bốc từ barem phía client, nhưng giữ lại nếu cần)
@login_required
@require_POST
def get_supplementary_for_session(request: HttpRequest, session_id: int) -> JsonResponse:
    session = get_object_or_404(ExamSession.objects.select_related("subject"), pk=session_id)
    _ensure_owner(session, request.user)

    pool = list(Question.objects.filter(subject=session.subject, is_supplementary=True))
    random.shuffle(pool)
    picked = pool[:2]

    items = [{"id": q.id, "question": q.question_text} for q in picked]
    return JsonResponse({"status": "ok", "items": items})


# Lưu một câu hỏi phụ (gọi trước khi finalize)
@login_required
@require_POST
def save_supplementary_result(request: HttpRequest) -> JsonResponse:
    """
    Body JSON:
      { session_id, question_text, transcript, score, max_score?, feedback?, analysis? }
    """
    data = _json_body(request)
    session_id = data.get("session_id")
    question_text = (data.get("question_text") or "").strip()
    transcript = data.get("transcript") or ""
    score = data.get("score")
    max_score = data.get("max_score")
    feedback = data.get("feedback")
    analysis = data.get("analysis")

    if session_id is None or not question_text or score is None:
        return HttpResponseBadRequest("Thiếu tham số.")

    session = get_object_or_404(ExamSession, pk=session_id)
    _ensure_owner(session, request.user)

    sr = SupplementaryResult.objects.create(
        session=session,
        question_text=question_text,
        transcript=transcript,
        score=float(score),
        max_score=float(max_score) if max_score is not None else 1.0,
        feedback=feedback,
        analysis=analysis,
    )
    return JsonResponse({"status": "ok", "supplementary_result_id": sr.id})


# Chốt phiên thi (tính tổng điểm, khóa session)
@login_required
@require_POST
def finalize_session_view(request: HttpRequest, session_id: int) -> JsonResponse:
    session = get_object_or_404(ExamSession, pk=session_id)
    _ensure_owner(session, request.user)

    main_avg, supp_sum, total = _compute_scores(session)
    if main_avg is None:
        return JsonResponse({"status": "error", "message": "Chưa có kết quả nào của phần chính."}, status=400)

    session.is_completed = True
    session.completed_at = timezone.now()
    session.final_score = total
    session.save(update_fields=["is_completed", "completed_at", "final_score"])

    return JsonResponse({"status": "success", "final_score": session.final_score})


# ===========================
# Legacy supplementary APIs (giữ nếu còn dùng ở nơi khác)
# ===========================

@login_required
@require_GET
def get_supplementary_questions_api(request: HttpRequest, session_id: int) -> JsonResponse:
    session = get_object_or_404(ExamSession.objects.select_related("subject"), pk=session_id)
    _ensure_owner(session, request.user)  # ✅ sửa: dùng request.user
    supp_qs = Question.objects.filter(subject=session.subject, is_supplementary=True)
    items = [{"id": q.id, "question": q.question_text} for q in supp_qs]
    return JsonResponse({"status": "ok", "items": items})

@login_required
@require_GET
def get_supplementary_questions(request: HttpRequest) -> JsonResponse:
    session_id = request.GET.get("session_id")
    subject_code = request.GET.get("subject_code")

    subject = None
    if session_id:
        session = get_object_or_404(ExamSession.objects.select_related("subject"), pk=int(session_id))
        _ensure_owner(session, request.user)
        subject = session.subject
    elif subject_code:
        subject = get_object_or_404(Subject, subject_code=subject_code)
    else:
        return HttpResponseBadRequest("Thiếu session_id hoặc subject_code.")

    pool = list(Question.objects.filter(subject=subject, is_supplementary=True))
    random.shuffle(pool)
    picked = [{"id": q.id, "question": q.question_text} for q in pool[:2]]
    return JsonResponse({"status": "ok", "items": picked})


# ===========================
# Profile (upload avatar)
# ===========================

@login_required
@require_POST
def update_profile_image(request: HttpRequest) -> JsonResponse:
    """
    Nhận FormData('profile_image') và lưu vào hồ sơ người dùng.
    Trả về JSON: { success: bool, image_url?: str, error?: str }
    """
    file_obj = request.FILES.get('profile_image')
    if not file_obj:
        return JsonResponse({"success": False, "error": "Thiếu file 'profile_image'."}, status=400)

    profile, _ = UserProfile.objects.get_or_create(user=request.user)

    # Xoá file cũ (tuỳ chọn)
    try:
        if getattr(profile, "profile_image", None) and getattr(profile.profile_image, "name", ""):
            profile.profile_image.delete(save=False)
    except Exception:
        pass

    profile.profile_image = file_obj
    profile.save()

    return JsonResponse({"success": True, "image_url": profile.profile_image.url})
