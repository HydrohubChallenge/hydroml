from django import forms
from .models import Project

class ProjectCreate(forms.ModelForm):
    class Meta:
        model = Project
        fields = [
            'name',
            'describe',
        ]

    def valid_form(self):
        obj=form.save(commit=False)
        obj.owner=self.request.user
        obj.save()
        return obj
