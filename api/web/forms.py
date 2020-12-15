from django import forms
from .models import Project

class ProjectCreate(forms.ModelForm):
    """
    def clean_dataset(self):
        uploaded_dataset = self.cleaned_data['dataset']
        # uploaded_dataset = uploaded_dataset.replace('\n','').replace('\r','')

        if uploaded_dataset:
            filename = uploaded_dataset.name
            if filename.endswith('.csv'):

                return uploaded_dataset
            else:
                raise forms.ValidationError(
                    "Please upload a .csv extension file only"
                )
        return uploaded_dataset
    """

    class Meta:
        model = Project
        fields = [
            'name',
            'description',
            'dataset',
        ]





