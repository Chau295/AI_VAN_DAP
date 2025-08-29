# qna/forms.py

from django import forms

class QuestionForm(forms.Form):
    question = forms.CharField(
        label='Câu hỏi của bạn',
        widget=forms.Textarea(attrs={'rows': 3, 'placeholder': 'Bạn muốn hỏi gì về Django?'})
    )