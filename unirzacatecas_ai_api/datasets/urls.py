from django.urls import path
from . import views

app_name = "datasets"
urlpatterns = [
    path("", view=views.DatasetAPIView.as_view(), name="datasets"),
]
