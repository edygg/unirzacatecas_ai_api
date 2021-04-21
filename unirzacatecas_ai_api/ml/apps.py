from django.apps import AppConfig
from django.utils.translation import gettext_lazy as _


class MlConfig(AppConfig):
    name = 'unirzacatecas_ai_api.ml'
    verbose_name = _("ML")
