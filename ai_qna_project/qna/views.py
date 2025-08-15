# ai_qna_project/qna/views.py
from __future__ import annotations

import json
import random
from typing import Tuple, Optional

from django.contrib.auth.decorators import login_required
from django.core.exceptions import PermissionDenied
from django.http import (
    JsonResponse,
    HttpRequest,
    HttpResponse,
    HttpResponseBadRequest,
)
from django.shortcuts import get_object_or_404, render
from django.utils import timezone
from django.views.decorators.http import require_POST, require_GET

from .models import (
    Subject,
    Question,
    ExamSession,
    ExamResult,
    SupplementaryResult,
    UserProfile,  # dùng cho cập nhật avatar
)

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
    if not mains:
        # Chưa có phần chính => coi như 0
        supp_sum = sum(sr.score or 0.0 for sr in SupplementaryResult.objects
                       .filter(session=session).only("score"))
        return None, supp_sum, 0.0

    main_avg = sum((er.score or 0.0) for er in mains) / len(mains)
    supp_sum = sum((sr.score or 0.0) for sr in SupplementaryResult.objects
                   .filter(session=session).only("score"))
    final_total = min(10.0, main_avg + supp_sum)
    return main_avg, supp_sum, final_total

# ===========================
# Pages
# ===========================

@login_required
def dashboard_view(request: HttpRequest) -> HttpResponse:
    """Trang chính đơn giản: liệt kê môn học & lịch sử gần đây."""
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
    return render(request, "qna/profile.html")

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
    ✅ LUỒNG MỚI:
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

    # ✅ BAREM cho câu hỏi phụ: lấy từ "câu hỏi chính còn lại" (loại trừ 3 câu đã chọn)
    remaining_main = main_pool_qs.exclude(id__in=[q.id for q in selected_questions]).values("id", "question_text")
    barem = [{"id": r["id"], "question": r["question_text"]} for r in remaining_main]

    return render(request, "qna/exam.html", {
        "subject": subject,
        "selected_questions": selected_questions,
        "session": session,
        "barem_json": json.dumps(barem, ensure_ascii=False),  # client dùng window.__BAREM__ để random 2 câu phụ
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
# Profile
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
        if profile.profile_image:
            profile.profile_image.delete(save=False)
    except Exception:
        pass

    profile.profile_image = file_obj
    profile.save()

    return JsonResponse({"success": True, "image_url": profile.profile_image.url})
