from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import HttpResponseRedirect
from django.shortcuts import render, redirect
from django.conf import settings
from .forms import ProjectCreate
from .models import Project

import os, io, csv
import pandas as pd

@login_required
def index(request):
    dash = Project.objects.filter(owner=request.user)
    return render(request, "web/dashboard.html", {"dash": dash})


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
        data_html = df.to_html()
        context = {'loaded_data': data_html, 'project': project_sel.name}
    except Project.DoesNotExist:
        return redirect("index")
    return render(request, "web/project_view.html", context)


@login_required
def update_project(request, project_id):
    project_id = int(project_id)
    try:
        project_sel = Project.objects.get(id=project_id)
    except Project.DoesNotExist:
        return redirect("index")
    project_form = ProjectCreate(request.POST or request.FILES or None, instance=project_sel)
    if project_form.is_valid():
        project_form.save()

        return redirect("index")
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
