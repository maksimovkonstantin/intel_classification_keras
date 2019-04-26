import sys; sys.path.append('..')
import keras
from classification_models import Classifiers


def make_model(network, input_shape, classes=6, predict_flag=0, drop_rate=0.3):
    classifier, _ = Classifiers.get(network)
    if predict_flag == 0:
        weights = 'imagenet'
    else:
        weights = None
    base_model = classifier(input_shape=input_shape, weights=weights, include_top=False)
    x = keras.layers.GlobalAveragePooling2D()(base_model.output)

    x = keras.layers.Dropout(drop_rate)(x)

    output = keras.layers.Dense(classes, activation='softmax')(x)
    model = keras.models.Model(inputs=[base_model.input], outputs=[output])

    return model
