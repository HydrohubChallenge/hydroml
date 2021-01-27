from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.shortcuts import render, redirect
from django.conf import settings
from django.utils.translation import get_language
from django.urls import reverse
from urllib.parse import urlencode
from .forms import ProjectCreate, LabelCreate
from .models import Project, Label, ProjectPrediction, Features
from .tasks import precipitation
from django.forms import inlineformset_factory
from django import forms

import json

import os
import pandas as pd


@login_required
def index(request):
    dash = Project.objects.filter(owner=request.user).order_by('created_at')
    current_language = get_language()
    content = {"dash": dash, "current_language": current_language}
    return render(request, "web/dashboard.html", content)


@login_required
def create(request):
    create = ProjectCreate()
    if request.method == "POST":
        create = ProjectCreate(request.POST, request.FILES)
        if create.is_valid():
            new_csv = Project(dataset=request.FILES['dataset'])
            new_csv.save()

            obj = create.save(commit=False)
            obj.owner = request.user
            obj.save()

            csv_delimiter = obj.delimiter
            csv_file = os.path.join(settings.MEDIA_ROOT, obj.dataset.name)
            file = open(csv_file, 'r')
            df = pd.read_csv(file, delimiter=csv_delimiter)

            if not 'label' in df.columns:
                df["label"] = " "
                df.to_csv(csv_file, sep=csv_delimiter, index=False)

            columnscsv = df.columns.values.tolist()

            for column in columnscsv:
                objectColumn = Features.objects.create(
                    column = column,
                    project = obj,
                )
                objectColumn.save()


            return redirect("index")
        else:
            content = {'form': create, "create_form": create}
            return render(request, "web/create_form.html", content)
    else:
        return render(request, "web/create_form.html", {"create_form": create, "method": 'create'})


def open_project(request, project_id):
    project_id = int(project_id)
    tab = request.GET.get('tab')
    try:
        project_sel = Project.objects.get(id=project_id)
        csv_delimiter = project_sel.delimiter
        csv_file = os.path.join(settings.MEDIA_ROOT, project_sel.dataset.name)
        file = open(csv_file, 'r')
        df = pd.read_csv(file, delimiter=csv_delimiter)
        df.fillna(0)
        data = df.values.tolist()
        columnscsv = df.columns.values.tolist()
        pages = round(len(df.index)/100)

        data = json.dumps(data)
        columns = json.dumps(columnscsv)

        labels = Label.objects.filter(project=project_id)

        predictions = ProjectPrediction.objects.filter(project=project_id)

        features = inlineformset_factory(Project, Features, fields=('type','column'), extra=0)

        if request.method == 'POST':
            formset = features(request.POST, instance=project_sel)
            if formset.is_valid():
                formset.save()
        else:
            formset = features(instance=project_sel)


        content = {
            'loaded_data': data,
            'columns': columns,
            'project': project_sel,
            'labels': labels,
            'predictions': predictions,
            'tab': tab,
            'columnscsv': columnscsv,
            'formset': formset,

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
            if bool(request.FILES.get('dataset', False)) == True:
                new_csv = Project(dataset=request.FILES['dataset'])
                new_csv.save()

            project_form.save()

            return redirect("index")

    else:
        project_form = ProjectCreate(instance=project_sel)

    return render(request, "web/create_form.html", {"create_form": project_form, "method": 'update'})


@login_required
def clone_project(request, project_id):
    project_id = int(project_id)
    try:
        project_sel = Project.objects.get(id=project_id)
        labels_sel = Label.objects.filter(project_id=project_id)
    except Project.DoesNotExist:
        return redirect("index")
    project_sel.pk = None
    new_name = project_sel.name + " (copy)"
    project_sel.name = new_name
    project_sel.save()
    for label in labels_sel:
        label.pk = None
        label.project_id = project_sel.id
        label.save()

    return redirect("index")


@login_required
def delete_project(request, project_id):
    project_id = int(project_id)
    try:
        project_sel = Project.objects.get(id=project_id)
        Label.objects.filter(project_id=project_id).delete()
        Features.objects.filter(project_id=project_id).delete()
    except Project.DoesNotExist:
        return redirect("index")
    project_sel.delete()
    return redirect("index")


@login_required
def create_label(request, project_id):
    create = LabelCreate()
    if request.method == "POST":
        create = LabelCreate(request.POST)
        if create.is_valid():
            obj = create.save(commit=False)
            obj.project_id = project_id
            obj.save()
            return redirect('open-project', project_id=project_id)
        else:
            content = {'form': create, "create_label": create}
            return render(request, "web/create_label.html", content)
    else:
        return render(request, "web/create_label.html",
                      {"create_label": create, "method": 'create', "project": Project.objects.get(id=project_id)})


@login_required
def update_label(request, project_id, label_id):
    label_id = int(label_id)
    try:
        label_sel = Label.objects.get(id=label_id)
    except Label.DoesNotExist:
        return redirect('open-project', project_id=project_id)
    if request.method == 'POST':
        label_form = LabelCreate(request.POST, instance=label_sel)

        if label_form.is_valid():
            label_form.save()

            return redirect('open-project', project_id=project_id)

    else:
        label_form = LabelCreate(instance=label_sel)

    return render(request, "web/create_label.html",
                  {"create_label": label_form, "method": 'update', "project": Project.objects.get(id=project_id)})


@login_required
def clone_label(request, project_id, label_id):
    label_id = int(label_id)
    try:
        label_sel = Label.objects.get(id=label_id)
    except Label.DoesNotExist:
        return redirect('open-project', project_id=project_id)
    label_sel.pk = None
    new_name = label_sel.name + " (copy)"
    label_sel.name = new_name
    label_sel.save()
    return redirect('open-project', project_id=project_id)


@login_required
def delete_label(request, project_id, label_id):
    label_id = int(label_id)
    try:
        label_sel = Label.objects.get(id=label_id)
    except Label.DoesNotExist:
        return redirect('open-project', project_id=project_id)
    label_sel.delete()
    return redirect('open-project', project_id=project_id)


@login_required
def train_project(request, project_id):
    project_id = int(project_id)
    prediction = ProjectPrediction.objects.create(
        project_id=project_id,
        status=False,
        confusion_matrix=0,
        accuracy=0,
        precision=0,
        recall=0,
        f1_score=0,
        pickle='-'
    )
    prediction.save()
    pred_id = prediction.id
    precipitation.delay(project_id, pred_id)
    messages.add_message(request, messages.SUCCESS, 'New training started')

    base_url = reverse('open-project', kwargs={'project_id': project_id})
    query_string = urlencode({'tab': "models"})
    url = '{}?{}'.format(base_url, query_string)

    return redirect(url)

@login_required
def delete_prediction(request, project_id, pred_id):
    pred_id = int(pred_id)
    try:
        pred_sel = ProjectPrediction.objects.get(id=pred_id)
        pred_sel.delete()
        base_url = reverse('open-project', kwargs={'project_id': project_id})
        query_string = urlencode({'tab': "models"})
        url = '{}?{}'.format(base_url, query_string)
        return redirect(url)
    except ProjectPrediction.DoesNotExist:
        # return redirect('open-project', project_id=project_id)
        return redirect('index')
