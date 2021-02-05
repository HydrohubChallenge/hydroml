from django import forms
from django.utils.translation import ugettext_lazy as _

from .models import Project, Label, Features



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
            'type',
            'delimiter',
            'dataset',
        ]
        widgets = {
            'delimiter': forms.Select(choices=OPTIONS, attrs={'class': 'form-control'}),
        }


    def clean_dataset(self):
        # csv_pattern = [
        #     'datetime',
        #     'station',
        #     'variable',
        #     'measurement',
        #     'quality'
        # ]

        csv_pattern = [
            'datetime',
            'central_farm',
            'chaa_creek',
            'hawkesworth_bridge',
            'santa_elena',
            'avg',
            'label'
        ]

        uploaded_dataset = self.cleaned_data["dataset"]
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


class LabelCreate(forms.ModelForm):

    class Meta:
        model = Label
        fields = [
            'name',
            'description',
            'color',
        ]
        widgets = {
            'color': forms.TextInput(attrs={'type': 'color'}),
        }

class FeatureInlineFormset(forms.BaseInlineFormSet):

    def clean(self):
        super().clean()
        target_columns_count = 0
        for form in self.forms:
            current_type = form.cleaned_data['type']

            if current_type == 1: #Target
                target_columns_count += 1

        if target_columns_count > 1:
            
            raise forms.ValidationError('Apenas uma coluna target é permitida por dataset')
