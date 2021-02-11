"""hydroml URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name = 'index'),
    path('create/', views.create, name='create-project'),
    path('update/<int:project_id>', views.update_project),
    path('clone/<int:project_id>', views.clone_project),
    path('delete/<int:project_id>', views.delete_project, name='delete-project'),
    path('project/<int:project_id>', views.open_project, name='open-project'),
    path('project/<int:project_id>/train/', views.train_project),
    path('project/<int:project_id>/create_label/', views.create_label),
    path('project/<int:project_id>/update/<int:label_id>', views.update_label),
    path('project/<int:project_id>/clone/<int:label_id>', views.clone_label),
    path('project/<int:project_id>/delete/<int:label_id>', views.delete_label),
    path('project/<int:project_id>/deletepred/<int:prediction_id>', views.delete_model),
    path('project/<int:project_id>/download/<int:prediction_id>', views.download_model),
    path('download_prediction/', views.download_prediction, name='download-prediction'),
    path('project/<int:project_id>/prediction/<int:prediction_id>', views.make_prediction),
    path('create_feature/<int:project_id>', views.create_feature, name='create-feature'),
]
