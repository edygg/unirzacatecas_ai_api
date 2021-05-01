from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status

from unirzacatecas_ai_api.core import views as core_views
from . import models as datasets_models
from . import serializers as datasets_serializers

# Hola Allen
class DatasetAPIView(core_views.GeneralAPIView):

    parser_classes = (MultiPartParser, FormParser)

    def get(self, request, format=None):
        response = [dataset.to_representation() for dataset in datasets_models.DatasetModel.objects.all()]
        return Response(response)

    def post(self, request, format=None):
        dataset_serializer = datasets_serializers.DatasetSerializer(data=request.data)

        if dataset_serializer.is_valid():
            dataset_serializer.save()
            return Response(dataset_serializer.data, status=status.HTTP_201_CREATED)
        else:
            return Response(dataset_serializer.errors, status=status.HTTP_400_BAD_REQUEST)