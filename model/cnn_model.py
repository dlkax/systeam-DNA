from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0

def create_model(num_classes: int, input_shape=(224,224,3), pretrained=True):
    """Cria o modelo CNN com EfficientNetB0 como backbone"""

    if pretrained:
        weights = 'imagenet'
    else:
        weights = None

    # Backbone EfficientNetB0
    base_model = EfficientNetB0(
        include_top=False,
        weights=weights,
        input_shape=input_shape,
        pooling='avg'  # GlobalAveragePooling no final
    )

    # Congelar o backbone se quiser fine-tuning gradual
    if pretrained:
        base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model
