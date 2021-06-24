import pandas as pd
from django.db import models
from unirzacatecas_ai_api.core import models as core_models


class DatasetModel(core_models.Auditable):
    original_csv_file = models.FileField(upload_to='datasets/', null=False, blank=False)
    normalized_csv_file = models.FileField(upload_to='datasets/', null=True, blank=True)

    def to_representation(self):
        dataset_csv_file = self.original_csv_file
        dataframe = pd.read_csv(dataset_csv_file.path)
        column_types = dataframe.dtypes.to_dict()

        def type_mapper(dtype):
            if dtype in ["float64", "int64"]:
                return "numeric"
            else:
                return "category"

        column_types_json = [dict(name=k, type=type_mapper(str(v))) for k, v in column_types.items()]


        return dict(
            id=self.id,
            original_csv_file=self.original_csv_file.url if self.original_csv_file else None,
            # normalized_csv_file=None if self.normalized_csv_file != None else self.normalized_csv_file,
            created_at=self.created_at,
            sample=dict(
                columns=column_types_json,
                data=dataframe.head(),
            )
        )
