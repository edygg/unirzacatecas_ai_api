from rest_framework.response import Response
from rest_framework import status

from unirzacatecas_ai_api.core import views as core_views
from . import serializers as ml_serializers


class TrainingAPIView(core_views.GeneralAPIView):

    def post(self, request, format=None, *args, **kwargs):
        print(f"El id de la url {self.kwargs['dataset_id']}")
        training_serializer = ml_serializers.TrainingSerializer(data=request.data)
        
        if training_serializer.is_valid():
            """
            TODO
            1. Buscar el dataset con el ID de la URL
            2. Revisar que venga al menos un feature
            3. Revisar que los features existan como campos en el dataset
            4. Revisar que el target exista en el dataset
            5. Revisar que tenemos al menos un modelo de entrenamiento en la lista
            6. Redireccionar en dataset a cada función asíncrona de entrenamiento
            """
            return Response(training_serializer.data, status=status.HTTP_202_ACCEPTED)
        else:
            return Response(training_serializer.errors, status=status.HTTP_400_BAD_REQUEST)