from django.db import models
from unirzacatecas_ai_api.core import models as core_models


class DatasetModel(core_models.Auditable):
    original_csv_file = models.FileField(upload_to='datasets/', null=False, blank=False)
    normalized_csv_file = models.FileField(upload_to='datasets/', null=True, blank=True)

    def to_representation(self):
        return dict(
            id=self.id,
            original_csv_file=self.original_csv_file.url if self.original_csv_file else None,
            # normalized_csv_file=None if self.normalized_csv_file != None else self.normalized_csv_file,
            created_at=self.created_at,
        )