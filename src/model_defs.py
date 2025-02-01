# model_defs.py
import tensorflow as tf
from tensorflow.keras import Model, Input, layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

def build_cnn(input_shape=(224,224,3), num_classes=4):
    inp = Input(shape=input_shape)
    x = layers.Conv2D(64, (3,3), activation='relu')(inp)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(64, (3,3), activation='relu')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(128, (3,3), activation='relu', name='cnn_last_conv')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes)(x)

    model = Model(inp, out, name="custom_cnn")
    model.compile(optimizer=Adam(),
                  loss=SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

def build_vgg16(input_shape=(224,224,3), num_classes=4):
    base = VGG16(include_top=False, weights='imagenet', input_shape=input_shape, pooling='max')
    x = base.output
    out = layers.Dense(num_classes)(x)
    model = Model(base.input, out, name="vgg16_custom")
    model.compile(optimizer=Adam(),
                  loss=SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model