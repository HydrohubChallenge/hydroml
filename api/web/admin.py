from django.contrib import admin

from .models import Project, ProjectLabel, ProjectFeature, ProjectPrediction

admin.site.register([Project, ProjectLabel, ProjectFeature, ProjectPrediction])
