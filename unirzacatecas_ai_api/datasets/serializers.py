from rest_framework import serializers
from . import models as datasets_models


class DatasetSerializer(serializers.ModelSerializer):
    class Meta:
        model = datasets_models.DatasetModel
        fields = ("original_csv_file", "normalized_csv_file")

    def to_representation(self, instance):
        return dict(
            id=instance.id,
            original_csv_file=instance.original_csv_file
        )
