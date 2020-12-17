from django import forms
from .models import Project

class ProjectCreate(forms.ModelForm):

    class Meta:
        model = Project
        OPTIONS = (
            (',','Comma'),
            (';','Dot-Comma'),
            ('\t','Tab'),
            (' ','Space'),
        )
        fields = [
            'name',
            'description',
            'delimiter',
            'dataset',
        ]
        widgets = {
            'delimiter': forms.Select(choices=OPTIONS, attrs={'class': 'form-control'}),
        }


    def clean_dataset(self):
        csv_pattern = [
            'datetime',
            'station',
            'variable',
            'measurement',
            'quality'
        ]

        uploaded_dataset = self.cleaned_data["dataset"]
        delimiter = self.cleaned_data.get("delimiter")

        if uploaded_dataset:
            filename = uploaded_dataset.name
            if filename.endswith('.csv'):
                reader = uploaded_dataset.readline().decode("utf-8").splitlines()
                headers = reader[0].split(delimiter)

                if len(headers) == 1:
                    raise forms.ValidationError(
                        'Wrong delimiter for the uploaded file.'
                    )

                for key in csv_pattern:
                    if key not in headers:
                        raise forms.ValidationError(
                            'CSV file must have the following columns: {0}.'.format(csv_pattern)
                        )

                return uploaded_dataset
            else:
                raise forms.ValidationError(
                    "Please upload a .csv extension file only"
                )
        return uploaded_dataset





