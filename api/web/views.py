from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import HttpResponseRedirect
from django.shortcuts import render, redirect
from django.conf import settings
from django.utils.translation import get_language
from .forms import ProjectCreate, LabelCreate
from .models import Project, Label

import os, io, csv
import pandas as pd

@login_required
def index(request):
    dash = Project.objects.filter(owner=request.user)
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
            return redirect("index")
        else:
            content = {'form': create, "create_form": create}
            return render(request, "web/create_form.html", content)
    else:
        return render(request, "web/create_form.html", {"create_form": create})


def open_project(request, project_id):
    project_id = int(project_id)
    try:
        project_sel = Project.objects.get(id=project_id)
        csv_delimiter = project_sel.delimiter
        csv_file = os.path.join(settings.MEDIA_ROOT, project_sel.dataset.name)
        file = open(csv_file, 'r')
        df = pd.read_csv(file, nrows=100, delimiter=csv_delimiter)

        if not 'label' in df.columns:
            df["label"] = ""
            df.to_csv(csv_file, sep=csv_delimiter, index=False)


        data_html = df.to_html()

        labels = Label.objects.filter(project=project_id)

        content = {'loaded_data': data_html, 'project': project_sel, 'labels': labels}

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
    if request.method =='POST':
        project_form = ProjectCreate(request.POST, request.FILES, instance=project_sel)

        if project_form.is_valid():
            if bool(request.FILES.get('dataset', False)) == True:
                new_csv = Project(dataset=request.FILES['dataset'])
                new_csv.save()

            project_form.save()

            return redirect("index")

    else:
        project_form = ProjectCreate(instance=project_sel)

    return render(request, "web/create_form.html", {"create_form": project_form})


@login_required
def delete_project(request, project_id):
    project_id = int(project_id)
    try:
        project_sel = Project.objects.get(id=project_id)
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
        return render(request, "web/create_label.html", {"create_label": create})


@login_required
def update_label(request, project_id, label_id):
    label_id = int(label_id)
    try:
        label_sel = Label.objects.get(id=label_id)
    except Label.DoesNotExist:
        return redirect('open-project', project_id=project_id)
    if request.method =='POST':
        label_form = LabelCreate(request.POST, instance=label_sel)

        if label_form.is_valid():
            label_form.save()

            return redirect('open-project', project_id=project_id)

    else:
        label_form = LabelCreate(instance=label_sel)

    return render(request, "web/create_label.html", {"create_label": label_form})


@login_required
def delete_label(request, project_id, label_id):
    label_id = int(label_id)
    try:
        label_sel = Label.objects.get(id=label_id)
    except Label.DoesNotExist:
        return redirect('open-project', project_id=project_id)
    label_sel.delete()
    return redirect('open-project', project_id=project_id)