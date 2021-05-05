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
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
# router.register(r'prediction', views.ApiDataViewSet)

urlpatterns = [
    path('', views.index, name='index'),
    path('create/', views.create, name='create-project'),
    path('update/<int:project_id>', views.update_project, name='update-project'),
    path('clone/<int:project_id>', views.clone_project),
    path('delete/<int:project_id>', views.delete_project, name='delete-project'),
    path('project/<int:project_id>/data/', views.data_tab, name='data-tab'),
    path('project/<int:project_id>/features/', views.features_tab, name='features-tab'),
    path('project/<int:project_id>/labels/', views.labels_tab, name='labels-tab'),
    path('project/<int:project_id>/models/', views.models_tab, name='models-tab'),
    path('project/<int:project_id>/parameters/', views.parameters_tab, name='parameters-tab'),
    path('project/<int:project_id>/train/', views.train_project, name='train-project'),
    path('project/<int:project_id>/create_label/', views.create_label, name='create-label'),
    path('project/<int:project_id>/update/<int:label_id>', views.update_label, name='update-label'),
    path('project/<int:project_id>/clone/<int:label_id>', views.clone_label, name='clone-label'),
    path('project/<int:project_id>/delete/<int:label_id>', views.delete_label, name='delete-label'),
    path('project/<int:project_id>/deletepred/<int:prediction_id>', views.delete_model, name='delete-model'),
    path('download/<int:prediction_id>', views.download_model, name='download-model'),
    path('download_prediction/<int:prediction_id>', views.download_prediction, name='download-prediction'),
    path('project/<int:project_id>/prediction/<int:prediction_id>', views.make_prediction, name='make-prediction'),
    path('features/<int:project_id>', views.create_feature, name='create-feature'),
    path('hyperparameters/<int:project_id>', views.create_parameters, name='create-parameters'),
    path('api/predict/', views.api_prediction),
]
