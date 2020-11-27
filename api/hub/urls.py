from django.urls import path
from . import views
from hydroml.settings import DEBUG, STATIC_ROOT, STATIC_URL
from django.conf.urls.static import static


urlpatterns = [
    path('', views.index, name = 'index'),
    path('create/', views.create, name='create-project'),
    path('update/<int:project_id>', views.update_project),
    path('delete/<int:project_id>', views.delete_project),
]


if DEBUG:
    urlpatterns += static(STATIC_URL, document_root = STATIC_ROOT)