# qna/views.py
from __future__ import annotations

import json
import random
from base64 import b64encode
from typing import Optional, List

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
from django.db.models import Avg
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
# Scoring rules for supplementary questions
# ===========================
SUPP_MAX_PER_QUESTION = 1.0   # mỗi câu phụ tối đa 1.0
SUPP_MAX_COUNT = 2            # tối đa 2 câu phụ
FINAL_CAP = 7.0               # tổng điểm cuối (nếu có điểm phụ) không vượt quá 7.0

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

def _compute_scores(session):
    """
    Trả về (main_avg_on_10, supp_sum_on_2, final_total_display_on_10).

    - Truy vấn trực tiếp qua ExamResult / SupplementaryResult để không phụ thuộc related_name.
    - Mỗi câu phụ clamp 0..1, chỉ lấy tối đa 2 câu cao nhất.
    - Nếu có điểm phụ => tổng điểm cuối cap 7.0.
      Nếu KHÔNG có điểm phụ => tổng = điểm TB phần chính (tối đa 10).
    """
    # Điểm phần chính
    mains = list(ExamResult.objects.filter(session=session).only("score"))
    if not mains:
        return None, 0.0, 0.0
    main_avg = sum((float(er.score) if er.score is not None else 0.0) for er in mains) / len(mains)

    # Điểm câu phụ
    supp_scores = list(
        SupplementaryResult.objects.filter(session=session).values_list("score", flat=True)
    )
    supp_scores = [max(0.0, min(float(s or 0.0), SUPP_MAX_PER_QUESTION)) for s in supp_scores]
    supp_scores.sort(reverse=True)
    supp_sum = sum(supp_scores[:SUPP_MAX_COUNT])

    # Tổng điểm hiển thị:
    # - Có điểm phụ -> cap 7.0
    # - Không có điểm phụ -> theo TB phần chính, tối đa 10
    if supp_sum > 0:
        final_total = min(FINAL_CAP, (main_avg or 0.0) + supp_sum)
    else:
        final_total = min(10.0, (main_avg or 0.0))

    return main_avg, supp_sum, final_total

def _dedupe_supp_for_display(qs) -> List[SupplementaryResult]:
    """
    Khử lặp câu phụ theo question_text để hiển thị lịch sử.
    Nếu một câu xuất hiện nhiều bản ghi, giữ bản có điểm cao nhất.
    Trả về tối đa 2 mục theo thứ tự điểm giảm dần.
    """
    best_by_text = {}
    for s in qs:
        key = (s.question_text or "").strip()
        cur = best_by_text.get(key)
        if cur is None or float(s.score or 0) > float(cur.score or 0):
            best_by_text[key] = s
    items = list(best_by_text.values())
    items.sort(key=lambda x: float(x.score or 0), reverse=True)
    return items[:SUPP_MAX_COUNT]

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
    # (email/khoa không bắt buộc – bạn có thể bỏ nếu muốn)
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
        if pwd:
            temp_user = User(username=username)
            try:
                validate_password(pwd, user=temp_user)
            except ValidationError as e:
                self.add_error("password", e)
        return cleaned

# (ĐÃ LOẠI BỎ form cập nhật Khoa & Email – chỉ hiển thị read-only trong profile)

def _try_set_profile_fields(
    profile: UserProfile,
    *,
    full_name: Optional[str] = None,
    class_name: Optional[str] = None,
    faculty: Optional[str] = None,
    student_id: Optional[str] = None,
) -> None:
    """Gán an toàn các field nếu tồn tại trong UserProfile."""
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
    Đăng ký:
    - Lưu full_name/class_name/student_id vào UserProfile.
    - Tự đăng nhập và chuyển về dashboard.
    """
    if request.user.is_authenticated:
        return redirect("qna:dashboard")

    if request.method == "POST":
        form = RegistrationForm(request.POST)
        if form.is_valid():
            full_name = form.cleaned_data["full_name"].strip()
            username = form.cleaned_data["username"].strip()
            class_name = form.cleaned_data["class_name"].strip()
            email = (form.cleaned_data.get("email") or "").strip()
            faculty = (form.cleaned_data.get("faculty") or "").strip()
            password = form.cleaned_data["password"]

            user = User.objects.create_user(username=username, password=password, email=email or None)
            try:
                user.first_name = full_name
                if email:
                    user.email = email
                user.save(update_fields=["first_name", "email"] if email else ["first_name"])
            except Exception:
                pass

            profile, _ = UserProfile.objects.get_or_create(user=user)
            _try_set_profile_fields(
                profile,
                full_name=full_name,
                class_name=class_name,
                faculty=faculty or None,
                student_id=username,
            )

            login(request, user)
            messages.success(request, "Đăng ký thành công. Chào mừng bạn!")
            return redirect("qna:dashboard")
    else:
        form = RegistrationForm()

    return render(request, "registration/register.html", {"form": form})

# ===========================
# Profile helpers
# ===========================
def _avatar_data_url_from_profile(profile: UserProfile) -> str:
    """
    Tạo data-url từ blob trong DB nếu có (ưu tiên).
    Fallback: nếu còn FileField `profile_image` -> dùng .url
    Cuối cùng: trả về ảnh mặc định.
    """
    # Ưu tiên blob trong DB
    blob = getattr(profile, "profile_image_blob", None)
    if blob:
        try:
            mime = getattr(profile, "profile_image_mime", "") or "image/jpeg"
            return f"data:{mime};base64,{b64encode(blob).decode('ascii')}"
        except Exception:
            pass

    # Fallback: nếu có FileField cũ
    try:
        if getattr(profile, "profile_image", None) and hasattr(profile.profile_image, "url"):
            return profile.profile_image.url
    except Exception:
        pass

    # Default
    return static("images/default_avatar.png")

def _collect_profile_data(user) -> dict:
    """
    Thu thập dữ liệu hồ sơ để hiển thị (read-only):
    - Họ và tên / Lớp / MSSV lấy từ UserProfile đã set khi đăng ký.
    - Avatar lấy từ blob trong DB -> data URL, fallback file/url hoặc ảnh mặc định.
    """
    profile, _ = UserProfile.objects.get_or_create(user=user)

    full_name = getattr(profile, "full_name", None) or (user.first_name or "")
    class_name = getattr(profile, "class_name", None) or ""
    faculty = getattr(profile, "faculty", "") or ""  # có thể không dùng nữa
    student_id = getattr(profile, "student_id", "") or user.username

    avatar_url = _avatar_data_url_from_profile(profile)

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
        # Cờ cho template: không hiển thị form Khoa/Email nữa
        "allow_edit_email_faculty": False,
    }

# ===========================
# Pages
# ===========================
@login_required
def dashboard_view(request: HttpRequest) -> HttpResponse:
    """Trang chủ: Chào mừng và liệt kê các môn thi."""
    subjects = Subject.objects.all().order_by("name")
    recent_sessions = (
        ExamSession.objects.filter(user=request.user)
        .select_related("subject")
        .order_by("-created_at")[:10]
    )

    # Lấy tên đầy đủ để hiển thị lời chào
    profile_data = _collect_profile_data(request.user)

    return render(request, "qna/dashboard.html", {
        "subjects": subjects,
        "recent_sessions": recent_sessions,
        "full_name": profile_data.get("full_name")
    })

@login_required
def profile_view(request: HttpRequest) -> HttpResponse:
    """
    Trang hồ sơ: Chỉ hiển thị thông tin (read-only) & nút upload avatar.
    - KHÔNG còn form cập nhật Khoa/Email ở đây.
    - Upload ảnh dùng endpoint update_profile_image (FormData).
    """
    ctx = _collect_profile_data(request.user)
    ctx["form"] = None
    return render(request, "qna/profile.html", ctx)

@login_required
def history_view(request: HttpRequest) -> HttpResponse:
    sessions = (
        ExamSession.objects.filter(user=request.user)
        .select_related("subject")
        .order_by("-created_at")
    )
    for s in sessions:
        m, supp, t = _compute_scores(s)
        s.main_avg = m or 0.0
        s.supp_sum = supp or 0.0
        s.final_total = t or 0.0  # hiển thị /10 (nếu có điểm phụ thì đã cap 7)
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
    supp_qs = SupplementaryResult.objects.filter(session=session).order_by("created_at")
    supp_results = _dedupe_supp_for_display(supp_qs)

    main_avg, supp_sum, final_total = _compute_scores(session)

    return render(request, "qna/history_detail.html", {
        "session": session,
        "results": results,
        "supp_results": supp_results,
        "main_avg": main_avg,
        "supp_sum": supp_sum,
        "final_total": final_total,
    })

# ===========================
# Exam flow
# ===========================
@login_required
def exam_view(request: HttpRequest, subject_code: str) -> HttpResponse:
    subject = get_object_or_404(Subject, subject_code=subject_code)
    main_pool_qs = Question.objects.filter(subject=subject, is_supplementary=False)

    selected_questions = list(main_pool_qs.order_by("?")[:3])

    session = ExamSession.objects.create(user=request.user, subject=subject)
    if selected_questions:
        try:
            session.questions.set(selected_questions)
        except Exception:
            pass

    remaining_main = main_pool_qs.exclude(
        id__in=[q.id for q in selected_questions]
    ).values("id", "question_text")
    barem = [{"id": r["id"], "question": r["question_text"]} for r in remaining_main]

    return render(request, "qna/exam.html", {
        "subject": subject,
        "selected_questions": selected_questions,
        "session": session,
        "barem_json": json.dumps(barem, ensure_ascii=False),
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

@login_required
@require_POST
def save_supplementary_result(request: HttpRequest) -> JsonResponse:
    """
    Body JSON:
      { session_id, question_text, transcript, score, max_score?, feedback?, analysis? }

    Quy tắc:
      - Tối đa 2 câu phụ mỗi phiên.
      - Nếu client/worker gửi điểm thang 10 -> scale về 1.0 (mỗi câu tối đa 1.0).
    """
    data = _json_body(request)
    session_id = data.get("session_id")
    question_text = (data.get("question_text") or "").strip()
    transcript = data.get("transcript") or ""
    raw_score = data.get("score")
    raw_max = data.get("max_score")
    feedback = data.get("feedback")
    analysis = data.get("analysis")

    if session_id is None or not question_text or raw_score is None:
        return HttpResponseBadRequest("Thiếu tham số.")

    session = get_object_or_404(ExamSession, pk=session_id)
    _ensure_owner(session, request.user)

    if SupplementaryResult.objects.filter(session=session).count() >= SUPP_MAX_COUNT:
        return JsonResponse({"status": "error", "message": f"Tối đa {SUPP_MAX_COUNT} câu hỏi phụ cho mỗi phiên."}, status=400)

    try:
        rs = float(raw_score)
    except (TypeError, ValueError):
        rs = 0.0
    try:
        rm = float(raw_max) if raw_max is not None else 10.0
    except (TypeError, ValueError):
        rm = 10.0
    if rm <= 0:
        rm = 10.0

    scaled = (rs / rm) * SUPP_MAX_PER_QUESTION
    score_0_to_1 = max(0.0, min(scaled, SUPP_MAX_PER_QUESTION))

    sr = SupplementaryResult.objects.create(
        session=session,
        question_text=question_text,
        transcript=transcript,
        score=score_0_to_1,               # 0..1
        max_score=SUPP_MAX_PER_QUESTION,  # 1.0
        feedback=feedback,
        analysis=analysis,
    )

    main_avg, supp_sum, final_total = _compute_scores(session)
    return JsonResponse({
        "status": "ok",
        "supplementary_result_id": sr.id,
        "main_avg": main_avg,
        "supp_sum": supp_sum,
        "final_total": final_total
    })

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
    session.final_score = total  # nếu có điểm phụ thì đã cap 7; nếu không, là TB chính (tối đa 10)
    session.save(update_fields=["is_completed", "completed_at", "final_score"])

    return JsonResponse({"status": "success", "final_score": session.final_score})

# ===========================
# Legacy supplementary APIs
# ===========================
@login_required
@require_GET
def get_supplementary_questions_api(request: HttpRequest, session_id: int) -> JsonResponse:
    session = get_object_or_404(ExamSession.objects.select_related("subject"), pk=session_id)
    _ensure_owner(session, request.user)
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
# Profile (upload avatar -> lưu CSDL)
# ===========================
@login_required
@require_POST
def update_profile_image(request: HttpRequest) -> JsonResponse:
    """
    Nhận FormData('profile_image') và lưu NHỊ PHÂN vào CSDL (UserProfile.profile_image_blob).
    Trả về JSON: { success, image_data_url?, error? }
    """
    file_obj = request.FILES.get('profile_image')
    if not file_obj:
        return JsonResponse({"success": False, "error": "Thiếu file 'profile_image'."}, status=400)

    # Giới hạn dung lượng cơ bản (vd 5MB)
    if file_obj.size and file_obj.size > 5 * 1024 * 1024:
        return JsonResponse({"success": False, "error": "Ảnh quá lớn (tối đa 5MB)."}, status=400)

    content = file_obj.read()
    mime = getattr(file_obj, "content_type", None) or "image/jpeg"

    profile, _ = UserProfile.objects.get_or_create(user=request.user)

    # Xoá file cũ ở FileField (nếu có) – tránh rác ổ đĩa
    try:
        if getattr(profile, "profile_image", None) and getattr(profile.profile_image, "name", ""):
            profile.profile_image.delete(save=False)
    except Exception:
        pass

    # Lưu blob vào DB
    if hasattr(profile, "profile_image_blob"):
        profile.profile_image_blob = content
        if hasattr(profile, "profile_image_mime"):
            profile.profile_image_mime = mime
        try:
            profile.save(update_fields=["profile_image_blob", "profile_image_mime"] if hasattr(profile, "profile_image_mime") else ["profile_image_blob"])
        except Exception:
            profile.save()
    else:
        return JsonResponse({"success": False, "error": "Thiếu field profile_image_blob trong UserProfile."}, status=500)

    data_url = f"data:{mime};base64,{b64encode(content).decode('ascii')}"
    return JsonResponse({"success": True, "image_data_url": data_url})
