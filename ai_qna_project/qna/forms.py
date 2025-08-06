# qna/forms.py

from django import forms
from django.contrib.auth.models import User
from .models import UserProfile

INPUT_CLASSES = 'appearance-none bg-gray-100 rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline'
HELP_TEXT_CLASSES = 'text-gray-500 text-xs mt-1'


class RegistrationForm(forms.Form):
    full_name = forms.CharField(
        label='Họ và tên',
        max_length=255,
        required=True,
        widget=forms.TextInput(attrs={
            'placeholder': 'Ví dụ: Nguyễn Văn A',
            'class': INPUT_CLASSES
        })
    )

    username = forms.CharField(
        label='Mã sinh viên (Tên đăng nhập)',
        max_length=150,
        required=True,
        widget=forms.TextInput(attrs={
            'placeholder': 'Nhập mã sinh viên của bạn',
            'class': INPUT_CLASSES
        })
    )

    class_name = forms.CharField(
        label='Lớp',
        max_length=100,
        required=True,
        widget=forms.TextInput(attrs={
            'placeholder': 'Ví dụ: 48K14.2',
            'class': INPUT_CLASSES
        })
    )

    password = forms.CharField(
        label="Mật khẩu",
        required=True,
        widget=forms.PasswordInput(attrs={
            'placeholder': 'Mật khẩu phải có ít nhất 8 ký tự',
            'class': INPUT_CLASSES
        })
    )

    password2 = forms.CharField(
        label="Nhập lại mật khẩu",
        required=True,
        widget=forms.PasswordInput(attrs={
            'placeholder': 'Nhập lại mật khẩu để xác nhận',
            'class': INPUT_CLASSES
        })
    )

    def clean_username(self):
        username = self.cleaned_data.get('username')
        if User.objects.filter(username=username).exists():
            raise forms.ValidationError("Mã sinh viên này đã được sử dụng để đăng ký.")
        return username

    def clean(self):
        cleaned_data = super().clean()
        password = cleaned_data.get("password")
        password2 = cleaned_data.get("password2")

        if password and password2 and password != password2:
            raise forms.ValidationError("Hai mật khẩu không khớp. Vui lòng nhập lại.")

        return cleaned_data

    def save(self):
        data = self.cleaned_data

        user = User.objects.create_user(
            username=data.get('username'),
            password=data.get('password')
        )

        UserProfile.objects.create(
            user=user,
            full_name=data.get('full_name'),
            class_name=data.get('class_name'),
            student_id=data.get('username')
        )

        return user


class QuestionForm(forms.Form):
    question = forms.CharField(
        label='Câu hỏi của bạn',
        widget=forms.Textarea(attrs={'rows': 3, 'placeholder': 'Bạn muốn hỏi gì về Django?'})
    )