from django.db import models
from datetime import datetime


class Auditable(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        abstract = True

