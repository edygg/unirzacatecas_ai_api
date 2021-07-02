from django.db import models
from django.contrib.postgres import fields as pg_fields

from unirzacatecas_ai_api.core import models as core_models


class Algorithm(core_models.Auditable):
    # Categorias o tipos de algoritmos
    REGRESSOR = "regressor"
    CLASSIFIER = "classifier"

    # Algoritmos soportados clasificación
    DECISION_TREE = "decision_tree"
    RANDOM_FOREST = "random_forest"
    LOGISTIC_REGRESSION = "logistic_regression"
    SVM = "svm"
    NEURAL_NETWORKS = "neural_networks"
    NAIVE_BAYES = "naive_bayes"

    # Algoritmos soportados regresión
    # DECISION_TREE = "decision_tree"
    # RANDOM_FOREST = "random_forest"
    LINEAR_REGRESSION = "linear_regression"
    SVR = "svr"
    # NEURAL_NETWORKS = "neural_networks"
    POLYNOMIAL_REGRESSION = "polynomial_regression"

    NAME_CHOICES = (
        (DECISION_TREE, "decision_tree"),
        (RANDOM_FOREST, "random_forest"),
        (LOGISTIC_REGRESSION, "logistic_regression"),
        (SVM, "svm"),
        (NEURAL_NETWORKS, "neural_networks"),
        (NAIVE_BAYES, "naive_bayes"),
        (LINEAR_REGRESSION, "linear_regression"),
        (SVR, "svr"),
        (POLYNOMIAL_REGRESSION, "polynomial_regression"),
    )

    CATETORY_CHOICES = (
        (REGRESSOR, "Regresión"),
        (CLASSIFIER, "Clasificación"),
    )

    name = models.CharField(max_length=100, choices=NAME_CHOICES)
    category = models.CharField(max_length=100, choices=CATETORY_CHOICES)
    settings = pg_fields.JSONField()

    def __str__(self):
        return f'{self.name}-{self.category}'


class TrainingRun(core_models.Auditable):
    STATUS_STARTED = "started"
    STATUS_FINISHED = "finished"
    STATUS_WITH_ERRORS = "with_errors"

    STATUS_CHOICES = (
        (STATUS_STARTED, "Iniciado"),
        (STATUS_FINISHED, "finalizado"),
        (STATUS_WITH_ERRORS, "with_errors")
    )

    email = models.CharField(max_length=100, default="")
    name = models.CharField(max_length=100, choices=Algorithm.NAME_CHOICES)
    category = models.CharField(max_length=100, choices=Algorithm.CATETORY_CHOICES)
    has_error = models.BooleanField(default=False)
    errors = models.TextField()
    status = models.CharField(max_length=100, choices=STATUS_CHOICES, default=STATUS_STARTED)
    result = pg_fields.JSONField()
