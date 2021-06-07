from config import celery_app
import pandas as pd
import numpy as np
import random

from unirzacatecas_ai_api.datasets import models as datasets_models
from . import utils

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
    # TODO Tomar este dato de los settings del paquete
    test_size = 0.40 


    return train_test_split(
        dataframe, 
        dataframe[packet["target"]["name"]], 
        test_size=test_size, 
        random_state=seed
    )


# Algoritmos de regresión
@celery_app.task()
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
    from sklearn.model_selection import GridSearchCV, train_test_split

    if not utils.validate_packet(packet):
        print("Error en el paquete")
        return
    
    dataset = utils.get_dataset(packet)

    if not dataset:
        print("No se encontró el dataset")
        return

    x_train, x_test, y_train, y_test = decision_tree_common(packet, dataset)

    training_regressor = DecisionTreeRegressor(random_state=1)
    training_regressor_params = {'max_depth':np.arange(1,50,2),'min_samples_leaf':np.arange(2,15)}
    gs_training_regressor = GridSearchCV(
        training_regressor, 
        training_regressor_params, 
        cv=3
    )

    gs_training_regressor.fit(x_train,y_train)
    a = gs_training_regressor.best_params_
    print(a)
    training_regressor.fit(x_train, y_train)

    predictions = gs_training_regressor.predict(x_test)
    print(predictions)
    

# Algoritmos de clasificación
@celery_app.task()
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
    from sklearn.model_selection import GridSearchCV, train_test_split

    if not utils.validate_packet(packet):
        print("Error en el paquete")
        return
    
    dataset = utils.get_dataset(packet)

    if not dataset:
        print("No se encontró el dataset")
        return

    x_train, x_test, y_train, y_test = decision_tree_common(packet, dataset)

    training_classifier = DecisionTreeClassifier()
    training_classifier_params = { 'max_depth':np.arange(1,50,2), 'min_samples_leaf':np.arange(2,15) }
    gs_training_classifier = GridSearchCV(
        training_classifier, 
        training_classifier_params, 
        cv=3
    )

    gs_training_classifier.fit(x_train,y_train)
    a = gs_training_classifier.best_params_
    print(a)
    training_classifier.fit(x_train, y_train)

    predictions = training_classifier.predict(x_test)
    print(predictions)