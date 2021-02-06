from django import forms
from django.utils.translation import ugettext_lazy as _

from .models import Project, ProjectLabel, ProjectFeature


class ProjectCreate(forms.ModelForm):
    class Meta:
        model = Project
        OPTIONS = (
            (',', 'Comma'),
            (';', 'Dot-Comma'),
            ('\t', 'Tab'),
            (' ', 'Space'),
        )
        fields = [
            'name',
            'description',
            'type',
            'delimiter',
            'dataset_file',
        ]
        widgets = {
            'delimiter': forms.Select(choices=OPTIONS, attrs={'class': 'form-control'}),
        }

    def clean_dataset_file(self):
        # HydroHUB
        # csv_pattern = [
        #     'datetime',
        #     'station',
        #     'variable',
        #     'measurement',
        #     'quality'
        # ]

        # Tadashi
        # csv_pattern = [
        #     'datetime',
        #     'central_farm',
        #     'chaa_creek',
        #     'hawkesworth_bridge',
        #     'santa_elena',
        #     'avg',
        #     'label'
        # ]

        # Reverton
        # csv_pattern = [
        #     'datetime',
        #     'station_id',
        #     'variable_id',
        #     'measured',
        #     'updated_at'
        # ]

        # Test
        csv_pattern = [
            'datetime'
        ]

        uploaded_dataset = self.cleaned_data["dataset_file"]
        delimiter = self.cleaned_data.get("delimiter")

        if uploaded_dataset:
            filename = uploaded_dataset.name
            if filename.endswith('.csv'):
                reader = uploaded_dataset.readline().decode("utf-8").splitlines()
                headers = reader[0].split(delimiter)

                if len(headers) <= 1:
                    raise forms.ValidationError(
                        _('Wrong delimiter for the uploaded file.')
                    )

                for key in csv_pattern:
                    if key not in headers:
                        raise forms.ValidationError(
                            _('CSV file must have the following columns: {0}.'.format(csv_pattern))
                        )

            else:
                raise forms.ValidationError(
                    _("Please upload a .csv extension file only")
                )
        return uploaded_dataset


class ProjectLabelCreate(forms.ModelForm):
    class Meta:
        model = ProjectLabel
        fields = [
            'name',
            'description',
            'color',
        ]
        widgets = {
            'color': forms.TextInput(attrs={'type': 'color'}),
        }


class ProjectFeatureInlineFormset(forms.BaseInlineFormSet):

    def clean(self):
        super().clean()
        target_columns_count = 0
        timestamp_columns_count = 0
        input_columns_count = 0

        for form in self.forms:
            current_type = form.cleaned_data['type']

            if current_type == ProjectFeature.Type.TARGET:
                target_columns_count += 1

            elif current_type == ProjectFeature.Type.TIMESTAMP:
                timestamp_columns_count += 1

            elif current_type == ProjectFeature.Type.INPUT:
                input_columns_count += 1

        if target_columns_count > 1:
            raise forms.ValidationError('You can have only one target column.')

        if timestamp_columns_count > 1:
            raise forms.ValidationError('You can have only one timestamp column.')

        if target_columns_count == 0:
            raise forms.ValidationError('You have to select one target column.')

        if timestamp_columns_count == 0:
            raise forms.ValidationError('You have to select one timestamp column.')

        if input_columns_count == 0:
            raise forms.ValidationError('You have to select at least one input column.')
