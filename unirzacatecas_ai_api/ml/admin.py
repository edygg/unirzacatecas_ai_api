from django.contrib import admin
from django.contrib.postgres import fields as pg_fields 
from jsoneditor.forms import JSONEditor

from . import models as ml_models


class AlgorithmAdmin(admin.ModelAdmin):
    formfield_overrides = {
        pg_fields.JSONField: { 'widget': JSONEditor },
    }


admin.site.register(ml_models.Algorithm, AlgorithmAdmin)
