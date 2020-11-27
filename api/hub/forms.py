from django import forms
from .models import Project

class ProjectCreate(forms.ModelForm):
    class Meta:
        model = Project
        fields = '__all__'