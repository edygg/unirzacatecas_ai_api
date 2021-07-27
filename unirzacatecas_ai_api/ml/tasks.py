from config import celery_app
import pandas as pd
import numpy as np
import random
from django.conf import settings

from . import utils
from . import models
from . import metrics
from . import docs

"""
packet = dict(
    dataset_id=3,
    features=[
        dict(name="instant", type="numeric"),
        dict(name="season", type="numeric"),
        dict(name="yr", type="numeric"),
        dict(name="mnth", type="numeric"),
        dict(name="hr", type="numeric"),
        dict(name="weekday", type="numeric"),
        dict(name="workingday", type="numeric"),
        dict(name="weathersit", type="numeric"),
        dict(name="temp", type="numeric"),
        dict(name="atemp", type="numeric"),
        dict(name="hum", type="numeric"),
        dict(name="windspeed", type="numeric"),
        dict(name="casual", type="numeric"),
        dict(name="registered", type="numeric"),
    ],
    target=dict(name="cnt", type="numeric"),
    settings=dict(),
)

"""

# Algoritmos de regresion
@celery_app.task(soft_time_limit=600, time_limit=700)
def decision_tree_regressor(packet):
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import train_test_split

    RANDOM_INT = 1  # random seed for the dataset
    PERFORMANCE_JOBS = -1  # -1 means using all processors

    training_run = models.TrainingRun.objects.create(email=packet["email"])
    print(f"Training Run: {training_run.id}")

    if not utils.validate_packet(packet):
        training_run.has_error = True
        training_run.errors = "Paquete mal formado, contacte al administrador del sistema"
        training_run.status = models.TrainingRun.STATUS_WITH_ERRORS
        training_run.save()
        return

    dataset = utils.get_dataset(packet)

    if not dataset:
        training_run.has_error = True
        training_run.errors = "No se encontró el dataset, contacte al administrador del sistema"
        training_run.status = models.TrainingRun.STATUS_WITH_ERRORS
        training_run.save()
        return

    try:
        algorithm_parameters = models.Algorithm.objects.get(
            name=models.Algorithm.DECISION_TREE,
            category=models.Algorithm.REGRESSOR
        ).settings
    except models.Algorithm.DoesNotExist:
        algorithm_parameters = {
        'max_depth':np.arange(1,50,2),
        'min_samples_leaf':np.arange(2,15)}

    hold_out = float(
        packet["validation_algorithm"]
            .get("options", dict(runs=5, training_size=0.30))
            .get("training_size")
    )

    dataset_csv_file = dataset.original_csv_file
    dataframe = pd.read_csv(dataset_csv_file.path)
    dataframe_columns = list(map(lambda feature: feature["name"], packet["features"]))
    dataframe_columns.append(packet["target"]["name"])

    print(dataframe.head())
    print(dataframe_columns)
    dataframe = dataframe[dataframe_columns]
    print(dataframe.head())

    target_column = packet["target"]["name"]

    # Data preprocessing
    X = dataframe.drop(columns=[target_column])
    y = dataframe[target_column].copy()

    # Data normalization
    standarScaler = StandardScaler()
    X = standarScaler.fit_transform(X)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=hold_out, random_state=RANDOM_INT)

    # Model definition
    model = DecisionTreeRegressor()
    cross_validation = float(
        packet["validation_algorithm"]
            .get("options", dict(runs=5, training_size=0.40))
            .get("runs")
    )
    
    try:
        # Model training
        grid_search_models = GridSearchCV(model, algorithm_parameters, n_jobs=PERFORMANCE_JOBS, cv=cross_validation)
        grid_search_models.fit(x_train, y_train)

        # Model validation
        predictions = grid_search_models.predict(x_test)
        print(grid_search_models.best_params_)
        print(predictions)

        report_html = metrics.regression_metrics(y_test, predictions)
        print(report_html)
        # training_run.result = dict(content=str(report_html))
        training_run.status = training_run.STATUS_FINISHED
        training_run.save()

        docs.generate_pdf_report(report_html, training_run)

    except Exception as error:
       # training_run.result = dict(content=str(error))
        print("Lancé una excepción")
        print(error)
        training_run.status = training_run.STATUS_WITH_ERRORS
        training_run.errors = str(error)
        training_run.save()

@celery_app.task(soft_time_limit=600, time_limit=700)
def neural_network_regressor(packet):
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import train_test_split

    RANDOM_INT = 1  # random seed for the dataset
    PERFORMANCE_JOBS = -1  # -1 means using all processors

    training_run = models.TrainingRun.objects.create(email=packet["email"])
    print(f"Training Run: {training_run.id}")

    if not utils.validate_packet(packet):
        training_run.has_error = True
        training_run.errors = "Paquete mal formado, contacte al administrador del sistema"
        training_run.status = models.TrainingRun.STATUS_WITH_ERRORS
        training_run.save()
        return

    dataset = utils.get_dataset(packet)

    if not dataset:
        training_run.has_error = True
        training_run.errors = "No se encontró el dataset, contacte al administrador del sistema"
        training_run.status = models.TrainingRun.STATUS_WITH_ERRORS
        training_run.save()
        return

    try:
        algorithm_parameters = models.Algorithm.objects.get(
            name=models.Algorithm.NEURAL_NETWORKS,
            category=models.Algorithm.REGRESSOR
        ).settings
    except models.Algorithm.DoesNotExist:
        algorithm_parameters = {
            #    'hidden_layer_sizes': [(100), (5,5,2), (5,10,15,20), (5,5,10,5,10), (5,5,10,5,10,5),],
            'hidden_layer_sizes': [(100), (5, 5, 2), ],
            'activation': ['relu'],
            'solver': ['adam'],
            'alpha': [0.0001],
            'learning_rate': ['constant'],
            'max_iter': [3000, 5000],
        }

    hold_out = float(
        packet["validation_algorithm"]
            .get("options", dict(runs=5, training_size=0.30))
            .get("training_size")
    )

    dataset_csv_file = dataset.original_csv_file
    dataframe = pd.read_csv(dataset_csv_file.path)
    dataframe_columns = list(map(lambda feature: feature["name"], packet["features"]))
    dataframe_columns.append(packet["target"]["name"])

    print(dataframe.head())
    print(dataframe_columns)
    dataframe = dataframe[dataframe_columns]
    print(dataframe.head())

    target_column = packet["target"]["name"]

    # Data preprocessing
    X = dataframe.drop(columns=[target_column])
    y = dataframe[target_column].copy()

    # Data normalization
    standarScaler = StandardScaler()
    X = standarScaler.fit_transform(X)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=hold_out, random_state=RANDOM_INT)

    # Model definition
    model = MLPRegressor()
    cross_validation = float(
        packet["validation_algorithm"]
            .get("options", dict(runs=5, training_size=0.40))
            .get("runs")
    )

    try:
        # Model training
        grid_search_models = GridSearchCV(model, algorithm_parameters, n_jobs=PERFORMANCE_JOBS, cv=cross_validation)
        grid_search_models.fit(x_train, y_train)

        # Model validation
        predictions = grid_search_models.predict(x_test)
        print(grid_search_models.best_params_)
        print(predictions)

        report_html = metrics.regression_metrics(y_test, predictions)
        print(report_html)
        # training_run.result = dict(content=str(report_html))
        training_run.status = training_run.STATUS_FINISHED
        training_run.save()

        docs.generate_pdf_report(report_html, training_run)

    except Exception as error:
       # training_run.result = dict(content=str(error))
        print("Lancé una excepción")
        print(error)
        training_run.status = training_run.STATUS_WITH_ERRORS
        training_run.errors = str(error)
        training_run.save()


# Algoritmos de clasificacion
@celery_app.task(soft_time_limit=600, time_limit=700)
def decision_tree_classifier(packet):
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import LabelEncoder
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import train_test_split

    RANDOM_INT = 1  # random seed for the dataset
    PERFORMANCE_JOBS = -1  # -1 means using all processors

    training_run = models.TrainingRun.objects.create(email=packet["email"])
    print(f"Training Run: {training_run.id}")

    if not utils.validate_packet(packet):
        training_run.has_error = True
        training_run.errors = "Paquete mal formado, contacte al administrador del sistema"
        training_run.status = models.TrainingRun.STATUS_WITH_ERRORS
        training_run.save()
        return

    dataset = utils.get_dataset(packet)

    if not dataset:
        training_run.has_error = True
        training_run.errors = "No se encontró el dataset, contacte al administrador del sistema"
        training_run.status = models.TrainingRun.STATUS_WITH_ERRORS
        training_run.save()
        return

    try:
        algorithm_parameters = models.Algorithm.objects.get(
            name=models.Algorithm.DECISION_TREE,
            category=models.Algorithm.CLASSIFIER
        ).settings
    except models.Algorithm.DoesNotExist:
        algorithm_parameters = {
        'max_depth':np.arange(1,50,2),
        'min_samples_leaf':np.arange(2,15)}

    hold_out = float(
        packet["validation_algorithm"]
            .get("options", dict(runs=5, training_size=0.30))
            .get("training_size")
    )

    dataset_csv_file = dataset.original_csv_file
    dataframe = pd.read_csv(dataset_csv_file.path)
    dataframe_columns = list(map(lambda feature: feature["name"], packet["features"]))
    dataframe_columns.append(packet["target"]["name"])

    print(dataframe.head())
    print(dataframe_columns)
    dataframe = dataframe[dataframe_columns]
    print(dataframe.head())

    target_column = packet["target"]["name"]

    # defining class column
    label_encoder = LabelEncoder()
    label_encoder.fit(dataframe[target_column])

    # Data preprocessing
    X = dataframe.drop(columns=[target_column])
    y = label_encoder.transform(dataframe[target_column].copy())

    # Data normalization
    standarScaler = StandardScaler()
    X = standarScaler.fit_transform(X)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=hold_out, random_state=RANDOM_INT)

    # Model definition
    model = DecisionTreeClassifier()
    cross_validation = float(
        packet["validation_algorithm"]
            .get("options", dict(runs=5, training_size=0.40))
            .get("runs")
    )

    try:
        # Model training
        grid_search_models = GridSearchCV(model, algorithm_parameters, n_jobs=PERFORMANCE_JOBS, cv=cross_validation)
        grid_search_models.fit(x_train, y_train)

        # Model validation
        predictions = grid_search_models.predict(x_test)
        print(grid_search_models.best_params_)
        print(predictions)

        report_html = metrics.classification_report(y_test, predictions)
        print(report_html)
        # training_run.result = dict(content=str(report_html))
        training_run.status = training_run.STATUS_FINISHED
        training_run.save()

        docs.generate_pdf_report(report_html, training_run)

    except Exception as error:
       # training_run.result = dict(content=str(error))
        print("Lancé una excepción")
        print(error)
        training_run.status = training_run.STATUS_WITH_ERRORS
        training_run.errors = str(error)
        training_run.save()

@celery_app.task(soft_time_limit=600, time_limit=700)
def neural_network_classifier(packet):
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import LabelEncoder
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import train_test_split

    RANDOM_INT = 1  # random seed for the dataset
    PERFORMANCE_JOBS = -1  # -1 means using all processors

    training_run = models.TrainingRun.objects.create(email=packet["email"])
    print(f"Training Run: {training_run.id}")

    if not utils.validate_packet(packet):
        training_run.has_error = True
        training_run.errors = "Paquete mal formado, contacte al administrador del sistema"
        training_run.status = models.TrainingRun.STATUS_WITH_ERRORS
        training_run.save()
        return

    dataset = utils.get_dataset(packet)

    if not dataset:
        training_run.has_error = True
        training_run.errors = "No se encontró el dataset, contacte al administrador del sistema"
        training_run.status = models.TrainingRun.STATUS_WITH_ERRORS
        training_run.save()
        return

    try:
        algorithm_parameters = models.Algorithm.objects.get(
            name=models.Algorithm.NEURAL_NETWORKS,
            category=models.Algorithm.CLASSIFIER
        ).settings
    except models.Algorithm.DoesNotExist:
        algorithm_parameters = {
        #    'hidden_layer_sizes': [(100), (5,5,2), (5,10,15,20), (5,5,10,5,10), (5,5,10,5,10,5),],
            'hidden_layer_sizes': [(100), (5,5,2),],
            'activation': ['relu'],
            'solver': ['adam'],
            'alpha': [0.0001],
            'learning_rate': ['constant'],
            'max_iter' : [3000,5000],
        }

    hold_out = float(
        packet["validation_algorithm"]
            .get("options", dict(runs=5, training_size=0.30))
            .get("training_size")
    )

    dataset_csv_file = dataset.original_csv_file
    dataframe = pd.read_csv(dataset_csv_file.path)
    dataframe_columns = list(map(lambda feature: feature["name"], packet["features"]))
    dataframe_columns.append(packet["target"]["name"])

    print(dataframe.head())
    print(dataframe_columns)
    dataframe = dataframe[dataframe_columns]
    print(dataframe.head())

    target_column = packet["target"]["name"]

    # defining class column
    label_encoder = LabelEncoder()
    label_encoder.fit(dataframe[target_column])

    # Data preprocessing
    X = dataframe.drop(columns=[target_column])
    y = label_encoder.transform(dataframe[target_column].copy())

    # Data normalization
    standarScaler = StandardScaler()
    X = standarScaler.fit_transform(X)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=hold_out, random_state=RANDOM_INT)

    # Model definition
    model = MLPClassifier()
    cross_validation = float(
        packet["validation_algorithm"]
            .get("options", dict(runs=5, training_size=0.40))
            .get("runs")
    )

    try:
        # Model training
        grid_search_models = GridSearchCV(model, algorithm_parameters, n_jobs=PERFORMANCE_JOBS, cv=cross_validation)
        grid_search_models.fit(x_train, y_train)

        # Model validation
        predictions = grid_search_models.predict(x_test)
        print(grid_search_models.best_params_)
        print(predictions)

        report_html = metrics.classification_report(y_test, predictions)
        print(report_html)
        # training_run.result = dict(content=str(report_html))
        training_run.status = training_run.STATUS_FINISHED
        training_run.save()

        docs.generate_pdf_report(report_html, training_run)

    except Exception as error:
       # training_run.result = dict(content=str(error))
        print("Lancé una excepción")
        print(error)
        training_run.status = training_run.STATUS_WITH_ERRORS
        training_run.errors = str(error)
        training_run.save()
