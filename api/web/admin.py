from django.contrib import admin
from .models import Project, Label

admin.site.register([Project, Label])