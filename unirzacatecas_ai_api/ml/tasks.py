from config import celery_app
import pandas as pd
import numpy as np
import random

from . import utils
from . import models

from . import metrics

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

# Utilitarios comunes para modelos de aprendizaje
def decision_tree_common(packet, dataset):
    from sklearn.model_selection import train_test_split

    dataset_csv_file = dataset.original_csv_file
    dataframe = pd.read_csv(dataset_csv_file.path)
    dataframe_columns = list(map(lambda feature: feature["name"], packet["features"]))
    dataframe_columns.append(packet["target"]["name"])

    print(dataframe.head())
    print(dataframe_columns)
    dataframe = dataframe[dataframe_columns]
    print(dataframe.head())
    seed = random.randint(1, 1000)
    print(packet["validation_algorithm"])
    test_size = float(
        packet["validation_algorithm"]
            .get("options", dict(runs=5, training_size=0.40))
            .get("training_size")
    )


    return train_test_split(
        dataframe,
        dataframe[packet["target"]["name"]],
        test_size=test_size,
        random_state=seed
    )


# Algoritmos de regresión
@celery_app.task(soft_time_limit=600, time_limit=700)
def decision_tree_regressor(packet):
    """
    Examples:

        packet = {
            "dataset_id": 1
            "features": [
                {
                    "name": "column1",
                    "type": "category",
                },
                {
                    "name": "column2",
                    "type": "numeric",
                }
            ],
            "target": {
                "name": "price",
		        "type": "numeric"
            },
            "settings": {

            }
        }
    """
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.model_selection import GridSearchCV

    training_run = models.TrainingRun.objects.create(email=packet["email"], result=dict(content=""))
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

    x_train, x_test, y_train, y_test = decision_tree_common(packet, dataset)

    try:
        algorithm_settings = models.Algorithm.objects.get(
            name=models.Algorithm.DECISION_TREE,
            category=models.Algorithm.REGRESSOR
        ).settings
    except models.Algorithm.DoesNotExist:
        algorithm_settings = {'max_depth':np.arange(1,50,2),'min_samples_leaf':np.arange(2,15)}

    try:
        training_regressor = DecisionTreeRegressor(random_state=1)
        training_regressor_params = algorithm_settings
        gs_training_regressor = GridSearchCV(
            training_regressor,
            training_regressor_params,
            cv=3
        )

        gs_training_regressor.fit(x_train,y_train)
        a = gs_training_regressor.best_params_
        training_regressor.fit(x_train, y_train)

        predictions = gs_training_regressor.predict(x_test)

        report_html = metrics.regression_metrics(y_test, predictions)
        training_run.result = dict(content=str(report_html))
        training_run.status = training_run.STATUS_FINISHED
        training_run.save()
    except Exception as error:
        training_run.result = dict(content=str())
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

    # Model training
    grid_search_models = GridSearchCV(model, algorithm_parameters, n_jobs=PERFORMANCE_JOBS, cv=cross_validation)
    grid_search_models.fit(x_train, y_train)

    # Model validation
    predictions = grid_search_models.predict(x_test)
    print(grid_search_models.best_params_)
    print(predictions)


# Algoritmos de clasificación
@celery_app.task(soft_time_limit=600, time_limit=700)
def decision_tree_classifier(packet):
    """
    URL: https://stackabuse.com/decision-trees-in-python-with-scikit-learn/

    Examples:

        packet = {
            "dataset_id": 1
            "features": [
                {
                    "name": "column1",
                    "type": "category",
                },
                {
                    "name": "column2",
                    "type": "numeric",
                }
            ],
            "target": {
                "name": "valueRange",
		        "type": "category"
            },
            "settings": {

            }
        }
    """
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import GridSearchCV

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

    x_train, x_test, y_train, y_test = decision_tree_common(packet, dataset)

    try:
        algorithm_settings = models.Algorithm.objects.get(
            name=models.Algorithm.DECISION_TREE,
            category=models.Algorithm.REGRESSOR
        ).settings
    except models.Algorithm.DoesNotExist:
        algorithm_settings = {'max_depth':np.arange(1,50,2),'min_samples_leaf':np.arange(2,15)}

    try:
        training_classifier = DecisionTreeClassifier()
        training_classifier_params = algorithm_settings
        gs_training_classifier = GridSearchCV(
            training_classifier,
            training_classifier_params,
            cv=3
        )

        gs_training_classifier.fit(x_train,y_train)
        a = gs_training_classifier.best_params_
        training_classifier.fit(x_train, y_train)

        predictions = training_classifier.predict(x_test)

    except Exception as error:
        training_run.result = dict(content=str())
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
            category=models.Algorithm.REGRESSOR
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

    # Model training
    grid_search_models = GridSearchCV(model, algorithm_parameters, n_jobs=PERFORMANCE_JOBS, cv=cross_validation)
    grid_search_models.fit(x_train, y_train)

    # Model validation
    predictions = grid_search_models.predict(x_test)
    print(grid_search_models.best_params_)
    print(predictions)
