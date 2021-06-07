from rest_framework import serializers


class TrainingFieldSerializer(serializers.Serializer):

    # Tipos de caraterísticas o campos
    NUMERIC = "numeric"
    CATEGORY = "category"

    TYPE_CHOICES = (
        (NUMERIC, "Número"),
        (CATEGORY, "Categórica"),
    )

    name = serializers.CharField(max_length=255)
    type = serializers.ChoiceField(choices=TYPE_CHOICES)


class AlgorithmSerializer(serializers.Serializer):
    name = serializers.CharField(max_length=255)
    options = serializers.JSONField()


class TrainingSerializer(serializers.Serializer):
    # Categorias o tipos de algoritmos
    REGRESSOR = "regressor"
    CLASSIFIER = "classifier"
    
    TYPE_CHOICES = (
        (REGRESSOR, "Regresión"),
        (CLASSIFIER, "Clasificación"),
    )

    email = serializers.EmailField()
    features = TrainingFieldSerializer(many=True, allow_null=False)
    target = TrainingFieldSerializer()
    type = serializers.ChoiceField(choices=TYPE_CHOICES)
    training_models = AlgorithmSerializer(many=True)
    validation_algorithm = AlgorithmSerializer()
