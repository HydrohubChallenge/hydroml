from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import HttpResponse
from django.shortcuts import render, redirect

from .forms import ProjectCreate
from .models import Project


@login_required
def index(request):
    dash = Project.objects.filter(owner=request.user)
    return render(request, "web/dashboard.html", {"dash": dash})


@login_required
def create(request):
    create = ProjectCreate()
    if request.method == "POST":
        create = ProjectCreate(request.POST)
        if create.is_valid():
            obj = create.save(commit=False)
            obj.owner = request.user
            obj.save()
            return redirect("index")
        else:
            return HttpResponse(
                """There is a problem in your form, reload the page <a href = "{{ url : 'index'}}">here</a>"""
            )
    else:
        return render(request, "web/create_form.html", {"create_form": create})


@login_required
def update_project(request, project_id):
    project_id = int(project_id)
    try:
        project_sel = Project.objects.get(id=project_id)
    except Project.DoesNotExist:
        return redirect("index")
    project_form = ProjectCreate(request.POST or None, instance=project_sel)
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
