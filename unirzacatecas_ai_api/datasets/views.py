from django.http import Http404
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status

from unirzacatecas_ai_api.core import views as core_views
from . import models as datasets_models
from . import serializers as datasets_serializers


class DatasetAPIView(core_views.GeneralAPIView):

    parser_classes = (MultiPartParser, FormParser)


    def get_object(self, pk):
        try:
            return datasets_models.DatasetModel.objects.get(pk=pk)
        except datasets_models.DatasetModel.DoesNotExist:
            raise Http404


    def get(self, request, format=None, *args, **kwargs):
        if self.kwargs.get("dataset_id"):
            response = self.get_object(self.kwargs.get("dataset_id")).to_representation()
            return Response(response)

        response = [dataset.to_representation() for dataset in datasets_models.DatasetModel.objects.all()]
        return Response(response)


    def post(self, request, format=None):
        dataset_serializer = datasets_serializers.DatasetSerializer(data=request.data)

        if dataset_serializer.is_valid():
            dataset_serializer.save()
            return Response(dataset_serializer.to_representation(), status=status.HTTP_201_CREATED)
        else:
            return Response(dataset_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
