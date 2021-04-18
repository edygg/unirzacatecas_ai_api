from rest_framework import serializers
from . import models as datasets_models


class DatasetSerializer(serializers.ModelSerializer):
    class Meta:
        model = datasets_models.DatasetModel
        fields = ("original_csv_file", "normalized_csv_file")