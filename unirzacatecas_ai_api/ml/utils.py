from unirzacatecas_ai_api.datasets import models as datasets_models
from . import constants as ml_constants
from . import models as ml_models
from . import tasks as ml_tasks

def validate_packet(packet):
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
    if not packet.get("dataset_id"):
        print("Falta el dataset id para procesar")
        return False

    if not packet.get("features"):
        print("Faltan los features para procesar")
        return False

    if len(packet.get("features")) < 1:
        print("Defina más de un features para procesar")
        return False

    if not packet.get("target"):
        print("Falta la variable objetivo para procesar")
        return False

    return True


def get_dataset(packet):
    try:
        dataset = datasets_models.DatasetModel.objects.get(id=packet["dataset_id"])
    except datasets_models.DatasetModel.DoesNotExist:
        return None
    else:
        return dataset


def get_empty_packet():
    return {
        ml_constants.MLPacket.DATASET_ID: 0,
        ml_constants.MLPacket.FEATURES: list(),
        ml_constants.MLPacket.TARGET: {
            ml_constants.MLPacket.Target.NAME: "",
            ml_constants.MLPacket.Target.TYPE: "",
        },
        ml_constants.MLPacket.SETTINGS: {

        },
    }


def algorithm_selector(algorithm_type, algorithm):
    if algorithm_type == ml_models.Algorithm.CLASSIFIER:
        if algorithm == ml_models.Algorithm.DECISION_TREE:
            return ml_tasks.decision_tree_classifier
        elif algorithm == ml_models.Algorithm.NEURAL_NETWORKS:
            return ml_tasks.neural_network_classifier
    elif algorithm_type == ml_models.Algorithm.REGRESSOR:
        if algorithm == ml_models.Algorithm.DECISION_TREE:
            return ml_tasks.decision_tree_regressor
        elif algorithm == ml_models.Algorithm.NEURAL_NETWORKS:
            return ml_tasks.neural_network_regressor


