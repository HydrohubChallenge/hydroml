from django import forms
from .models import Project

import csv, io

class ProjectCreate(forms.ModelForm):

    def clean_dataset(self):
        csv_pattern = ['datetime','measured','station_id','variable_id','updated_at']

        uploaded_dataset = self.cleaned_data['dataset']

        if uploaded_dataset:
            filename = uploaded_dataset.name
            if filename.endswith('.csv'):
                reader = uploaded_dataset.readline().decode("utf-8").splitlines()
                headers = reader[0].split(',')
                for key in csv_pattern:
                    if key not in headers:
                        raise forms.ValidationError(
                            'CSV file must have the following columns: {0}'.format(csv_pattern)
                        )

                return uploaded_dataset
            else:
                raise forms.ValidationError(
                    "Please upload a .csv extension file only"
                )
        return uploaded_dataset


    class Meta:
        model = Project
        fields = [
            'name',
            'description',
            'dataset',
        ]





