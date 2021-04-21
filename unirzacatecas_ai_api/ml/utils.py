from unirzacatecas_ai_api.datasets import models as datasets_models

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