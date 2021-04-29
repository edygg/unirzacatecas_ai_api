from config import celery_app
import pandas as pd
import numpy as np
import random

from unirzacatecas_ai_api.datasets import models as datasets_models
from . import utils

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
    

    dataset_csv_file = dataset.original_csv_file
    dataframe = pd.read_csv(dataset_csv_file.path)
    dataframe_columns = list(map(lambda feature: feature["name"], packet["features"]))
    
    # dataframe.columns = dataframe_columns
    print(dataframe_columns)
    print(dataframe.head())
    seed = random.randint(1, 1000)
    test_size = 0.40 


    x_train, x_test, y_train, y_test = train_test_split(
        dataframe.drop(columns=["dteday"]), 
        dataframe[packet["target"]["name"]], 
        test_size=test_size, 
        random_state=seed
    )

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
    