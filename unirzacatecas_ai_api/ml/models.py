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