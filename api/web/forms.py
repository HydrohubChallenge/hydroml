from django import forms
from .models import Project
from .widgets import BinaryFileInput

class ProjectCreate(forms.ModelForm):
    class Meta:
        model = Project
        fields = [
            'name',
            'description',
            'dataset',
        ]
        widgets = {
            'dataset': BinaryFileInput(),
        }

