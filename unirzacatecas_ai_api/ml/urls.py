from django.urls import path
from . import views

app_name = "ml"
urlpatterns = [
    path("<int:dataset_id>/training-model/", view=views.TrainingAPIView.as_view(), name="datasets"),
]