from django.shortcuts import render, redirect


def index(request):

    
    return render(request, 'index.html', {})


def login(request):
    return render(request, 'login.html', {})

def blank(request):
    return redirect('/login/')