import json
import json as simplejson
import os, tempfile
import pickle

import numpy as np

import pandas as pd
from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.forms import inlineformset_factory
from django.http import FileResponse, HttpResponseRedirect, HttpResponse
from django.shortcuts import render, redirect
from django.utils.translation import get_language

from .forms import ProjectCreate, ProjectLabelCreate, ProjectFeatureInlineFormset, ProjectPredictionUploadFile
from .models import Project, ProjectLabel, ProjectPrediction, ProjectFeature
from .tasks import train_precipitation_prediction, train_water_level_prediction

from keras.models import load_model


@login_required
def index(request):
    projects = Project.objects.filter(owner=request.user).order_by('created_at')
    current_language = get_language()
    content = {"projects": projects, "current_language": current_language}
    return render(request, "web/dashboard.html", content)


@login_required
def create(request):
    create_project_form = ProjectCreate()
    if request.method == "POST":
        create_project_form = ProjectCreate(request.POST, request.FILES)
        if create_project_form.is_valid():
            obj = create_project_form.save(commit=False)
            obj.owner = request.user
            obj.save()

            csv_delimiter = obj.delimiter
            csv_file = os.path.join(settings.MEDIA_ROOT, obj.dataset_file.name)
            file = open(csv_file, 'r')
            df = pd.read_csv(file, delimiter=csv_delimiter)

            if 'label' not in df.columns:
                df["label"] = " "
                df.to_csv(csv_file, sep=csv_delimiter, index=False)

            columns_csv = df.columns.values.tolist()

            for column in columns_csv:
                project_feature = ProjectFeature.objects.create(
                    column=column,
                    project=obj,
                )
                project_feature.save()

            return redirect("create-feature", project_id=obj.id)
        else:
            content = {"create_form": create_project_form}
            return render(request, "web/create_form.html", content)
    else:
        return render(request, "web/create_form.html", {"create_form": create_project_form, "method": 'create'})


def open_project(request, project_id):
    project_id = int(project_id)
    tab = request.GET.get('tab')
    try:
        project_sel = Project.objects.get(id=project_id)
        csv_delimiter = project_sel.delimiter
        csv_file = os.path.join(settings.MEDIA_ROOT, project_sel.dataset_file.name)
        file = open(csv_file, 'r')
        df = pd.read_csv(file, delimiter=csv_delimiter)
        df.fillna('NaN', inplace=True)
        data = df.values.tolist()
        columnscsv = df.columns.values.tolist()

        data = json.dumps(data)
        columns = json.dumps(columnscsv)

        labels = ProjectLabel.objects.filter(project=project_id)

        predictions = ProjectPrediction.objects.filter(project=project_id)

        project_feature_formset_factory = inlineformset_factory(Project, ProjectFeature, fields=(
            'type', 'column'), extra=0, formset=ProjectFeatureInlineFormset)

        if request.method == 'POST':
            project_feature_formset = project_feature_formset_factory(
                request.POST, instance=project_sel)
            if project_feature_formset.is_valid():
                project_feature_formset.save()
        else:
            project_feature_formset = project_feature_formset_factory(
                instance=project_sel)

        try:
            project_features = ProjectFeature.objects.filter(project_id=project_id)

            input = []
            timestamp = None
            for project_feature in project_features:
                if project_feature.type == ProjectFeature.Type.TIMESTAMP:
                    timestamp = project_feature.column
                elif project_feature.type == ProjectFeature.Type.INPUT:
                    input.append(project_feature.column)

            input_size = len(input)

            categories = list(df[timestamp])

            values_list = []

            for c in input:
                values = list(df[c])
                values_list.append(values)

            json_list = simplejson.dumps(values_list)

            content = {
                'loaded_data': data,
                'columns': columns,
                'project': project_sel,
                'labels': labels,
                'predictions': predictions,
                'tab': tab,
                'columnscsv': columnscsv,
                'project_feature_formset': project_feature_formset,
                'categories': categories,
                'values': json_list,
                'input_size': input_size,
                'input_columns': input,
            }
        except ProjectFeature.DoesNotExist:
            content = {
                'loaded_data': data,
                'columns': columns,
                'project': project_sel,
                'labels': labels,
                'predictions': predictions,
                'tab': tab,
                'columnscsv': columnscsv,
                'project_feature_formset': project_feature_formset,
            }

    except Project.DoesNotExist:
        return redirect("index")

    return render(request, "web/project_view.html", content)


@login_required
def update_project(request, project_id):
    project_id = int(project_id)

    try:
        project_sel = Project.objects.get(id=project_id)
    except Project.DoesNotExist:
        return redirect("index")

    if request.method == 'POST':
        project_form = ProjectCreate(request.POST, request.FILES, instance=project_sel)

        if project_form.is_valid():
            project_form.save()
            return redirect("index")

    else:
        project_form = ProjectCreate(instance=project_sel)

    return render(request, "web/create_form.html", {"create_form": project_form, "method": 'update'})


@login_required
def clone_project(request, project_id):
    project_id = int(project_id)

    try:
        project = Project.objects.get(id=project_id)
        project_labels = ProjectLabel.objects.filter(project_id=project_id)
        project_features = ProjectFeature.objects.filter(project_id=project_id)
    except Project.DoesNotExist:
        return redirect("index")

    project.id = None
    project.pk = None
    new_name = project.name + " (copy)"
    project.name = new_name
    project.save()

    for project_label in project_labels:
        project_label.id = None
        project_label.pk = None
        project_label.project_id = project.id
        project_label.save()

    for project_feature in project_features:
        project_feature.id = None
        project_feature.pk = None
        project_feature.project_id = project.id
        project_feature.save()

    return redirect("index")


@login_required
def delete_project(request, project_id):
    project_id = int(project_id)
    try:
        project_sel = Project.objects.get(id=project_id)
        ProjectLabel.objects.filter(project_id=project_id).delete()
        ProjectFeature.objects.filter(project_id=project_id).delete()
        ProjectPrediction.objects.filter(project_id=project_id).delete()
    except Project.DoesNotExist:
        return redirect("index")
    project_sel.delete()
    return redirect("index")


@login_required
def create_label(request, project_id):
    create_label_form = ProjectLabelCreate()
    if request.method == "POST":
        create_label_form = ProjectLabelCreate(request.POST)
        if create_label_form.is_valid():
            obj = create_label_form.save(commit=False)
            obj.project_id = project_id
            obj.save()
            return redirect('open-project', project_id=project_id)
        else:
            content = {"create_label": create_label_form}
            return render(request, "web/create_label.html", content)
    else:
        return render(request, "web/create_label.html",
                      {"create_label": create_label_form, "method": 'create',
                       "project": Project.objects.get(id=project_id)})


@login_required
def update_label(request, project_id, label_id):
    label_id = int(label_id)
    try:
        project_label = ProjectLabel.objects.get(id=label_id)
    except ProjectLabel.DoesNotExist:
        return redirect('open-project', project_id=project_id)
    if request.method == 'POST':
        label_form = ProjectLabelCreate(request.POST, instance=project_label)

        if label_form.is_valid():
            label_form.save()

            return redirect('open-project', project_id=project_id)

    else:
        label_form = ProjectLabelCreate(instance=project_label)

    return render(request, "web/create_label.html",
                  {"create_label": label_form, "method": 'update', "project": Project.objects.get(id=project_id)})


@login_required
def clone_label(request, project_id, label_id):
    try:
        project_label = ProjectLabel.objects.get(id=int(label_id))
    except ProjectLabel.DoesNotExist:
        return redirect('open-project', project_id=project_id)
    project_label.id = None
    project_label.pk = None
    new_name = project_label.name + " (copy)"
    project_label.name = new_name
    project_label.save()
    return redirect('open-project', project_id=project_id)


@login_required
def delete_label(request, project_id, label_id):
    try:
        ProjectLabel.objects.get(id=int(label_id)).delete()
    except ProjectLabel.DoesNotExist:
        return redirect('open-project', project_id=project_id)
    return redirect('open-project', project_id=project_id)


@login_required
def train_project(request, project_id):
    project_id = int(project_id)
    prediction = ProjectPrediction.objects.create(
        project_id=project_id,
        status=2,
        confusion_matrix=0,
        accuracy=0,
        precision=0,
        recall=0,
        f1_score=0,
    )
    prediction.save()

    if Project.objects.get(id=project_id).type == Project.Type.RAINFALL:
        train_precipitation_prediction.delay(project_id, prediction.id)

    elif Project.objects.get(id=project_id).type == Project.Type.WATER_LEVEL:
        train_water_level_prediction.delay(project_id, prediction.id)

    messages.add_message(request, messages.SUCCESS, 'New training started')

    return redirect('open-project', project_id=project_id)


@login_required
def delete_model(request, project_id, prediction_id):
    try:
        ProjectPrediction.objects.get(id=int(prediction_id)).delete()
    except ProjectPrediction.DoesNotExist:
        return redirect('index')

    return redirect('open-project', project_id=project_id)


@login_required
def download_model(request, project_id, prediction_id):
    try:
        project_prediction = ProjectPrediction.objects.get(id=int(prediction_id))
    except ProjectPrediction.DoesNotExist:
        return redirect('index')

    file_name = project_prediction.serialized_prediction_file.name
    response = FileResponse(open(file_name, 'rb'))
    return response


@login_required
def create_feature(request, project_id):
    try:
        project = Project.objects.get(id=int(project_id))
    except Project.DoesNotExist:
        return redirect("index")

    project_feature_formset_factory = inlineformset_factory(Project, ProjectFeature, fields=(
        'type', 'column'), extra=0, formset=ProjectFeatureInlineFormset)

    if request.method == 'POST':
        project_feature_formset = project_feature_formset_factory(
            request.POST, instance=project)
        if project_feature_formset.is_valid():
            project_feature_formset.save()
            return redirect("index")
    else:
        project_feature_formset = project_feature_formset_factory(
            instance=project)

    content = {
        'project_feature_formset': project_feature_formset
    }

    return render(request, "web/create_feature.html", content)


@login_required
def download_prediction(request):
    file_name = 'prediction.csv'
    file_path = os.path.join(tempfile.gettempdir(), file_name)
    if os.path.isfile(file_path):
        response = FileResponse(open(file_path, 'rb'))
        return response
    else:
        return redirect('index')

@login_required
def make_prediction(request, project_id, prediction_id):
    file_name = 'prediction.csv'
    file_path = os.path.join(tempfile.gettempdir(), file_name)
    number_rows = 0
    number_success = 0
    if request.method == 'POST':
        form = ProjectPredictionUploadFile(request.POST, request.FILES, project_id=project_id)
        if form.is_valid():
            file = request.FILES['file']
            export_result, number_rows, number_success = handle_uploaded_file(file, project_id, prediction_id)

            export_result.to_csv(path_or_buf=file_path)

    else:
        form = ProjectPredictionUploadFile()

    try:
        prediction = ProjectPrediction.objects.get(id=int(prediction_id))
    except ProjectPrediction.DoesNotExist:
        return redirect("index")

    content = {
        'form': form,
        'prediction': prediction,
        'number_rows': number_rows,
        'number_success': number_success,
    }

    return render(request, "web/make_prediction.html", content)


def handle_uploaded_file(file, project_id, prediction_id):
    project_id = int(project_id)
    prediction_id = int(prediction_id)

    try:
        project_sel = Project.objects.get(id=project_id)
    except Project.DoesNotExist:
        return redirect("index")

    try:
        project_prediction = ProjectPrediction.objects.get(id=int(prediction_id))
    except ProjectPrediction.DoesNotExist:
        return redirect('index')

    project_features = ProjectFeature.objects.filter(project_id=project_id)

    model_file = project_prediction.serialized_prediction_file.name

    df = pd.read_csv(file, delimiter=project_sel.delimiter, parse_dates=["datetime"])

    input_column_names = []

    for project_feature in project_features:
        if project_feature.type == ProjectFeature.Type.INPUT:
            input_column_names.append(project_feature.column)
        elif project_feature.type == ProjectFeature.Type.TIMESTAMP:
            input_column_names.append(project_feature.column)

    data_prediction = df[input_column_names]

    # if it's a Rainfall Project (1) import pickle
    # if it's a Water Level Project (2) import keras model
    if project_sel.type == 1:
        # Static prediction
        df_day = data_prediction.groupby([pd.Grouper(key="datetime", freq="2h"), "station", "station_id"]).sum()
        df_day.reset_index(inplace=True)
        df_day.groupby(['station']).agg({'datetime': [np.min, np.max]})

        data = df_day[['station', 'measured', 'datetime']].pivot(index='datetime', columns='station', values='measured')
        data.dropna(inplace=True)
        data.sort_index(inplace=True)

        def create_data_classification(df, columns, target, threshold):
            columns = columns[columns != target]

            df['avg'] = df[columns].mean(axis=1)

            df.loc[df[target] > df['avg'] * (1 + threshold), 'label'] = 0
            df.loc[df[target] <= df['avg'] * (1 + threshold), 'label'] = 1

            df = df.astype({'label': np.int})

            print(df.head())
            return df

        data = create_data_classification(data, np.array(['hawkesworth_bridge']), 'santa_elena', 0.3)

        X_test = data[['central_farm', 'chaa_creek', 'hawkesworth_bridge', 'santa_elena']]

        number_rows = len(X_test.index)
        loaded_model = pickle.load(open(model_file, 'rb'))
        prediction = loaded_model.predict(X_test)
        prediction = pd.Series(prediction, name='prediction')
        X_test.reset_index(inplace=True)
        export_df = pd.concat([X_test, prediction], axis=1)
        number_success = (prediction.values == 1).sum()

    elif project_sel.type == 2:
        loaded_model = load_model(model_file)

        for column in X_test.columns:
            print(X_test[column].dtype)
            if X_test[column].dtype == np.float64:
                X_test[column] = np.asarray(X_test[column]).astype(np.float32)
                print(X_test[column].dtype)
            elif X_test[column].dtype == np.object:
                X_test[column] = X_test[column].astype('|S')

        print(X_test.dtypes)

        loaded_model.predict(X_test)
        prediction = pd.Series(prediction, name='prediction')
        export_df = pd.concat([df, prediction], axis=1)

    return export_df, number_rows, number_success
