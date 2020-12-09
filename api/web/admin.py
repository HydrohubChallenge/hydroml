from django.contrib import admin
from django.db import models
from .models import Project, Label
from .widgets import BinaryFileInput


class CustomModelAdmin(admin.ModelAdmin):
    formfield_overrides = {
        models.BinaryField: {'widget': BinaryFileInput()}
    }

admin.site.register([Project, Label])