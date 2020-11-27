from django.shortcuts import render, redirect
from .models import Project
from .forms import ProjectCreate
from django.http import HttpResponse


def index(request):
    dash = Project.objects.all()
    return render(request, 'hub/dashboard.html',{'dash': dash})


def create(request):
    create = ProjectCreate()
    if request.method == 'POST':
        create = ProjectCreate(request.POST)
        if create.is_valid():
            create.save()
            return redirect('index')
        else:
            return HttpResponse("""There is a problem in your form, reload the page <a href = "{{ url : 'index'}}">here</a>""")
    else:
        return render(request, 'hub/create_form.html', {'create_form': create})


def update_project(request, project_id):
    project_id = int(project_id)
    try:
        project_sel = Project.objects.get(id = project_id)
    except Project.DoesNotExist:
        return redirect('index')
    project_form = ProjectCreate(request.POST or None, instance=project_sel)
    if project_form.is_valid():
        project_form.save()
        return redirect('index')
    return render(request, 'hub/create_form.html', {'create_form':project_form})


def delete_project(request, project_id):
    project_id = int(project_id)
    try:
        project_sel = Project.objects.get(id = project_id)
    except Project.DoesNotExist:
        return redirect('index')
    project_sel.delete()
    return redirect('index')
