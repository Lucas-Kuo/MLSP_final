import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras import regularizers
import os
import numpy as np
import config

def build_eff_model():
    base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2M(include_top=False, weights='imagenet',
                                                                      input_shape=config.input_shape,)

    base_model.trainable = True

    # Note the rescaling layer. These layers have pre-defined inference behavior.
    data_augmentation = keras.Sequential(
        [
            layers.Rescaling(scale=1.0 / 255),
            layers.RandomCrop(config.image_size, config.image_size),
            layers.RandomFlip("horizontal"),
        ],
        name="data_augmentation",
    )

    inputs = tf.keras.Input(shape=config.input_shape)
    label = tf.keras.Input(shape=(config.num_classes,))
    x = data_augmentation(inputs)
    x = base_model(x, training=False)  # set training to False to avoid BN training
    x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    output = tf.keras.layers.Dense(config.num_classes, activation="softmax",
                                    activity_regularizer=tf.keras.regularizers.L2(0.1))(x)
    model = tf.keras.Model(inputs, output)

    fine_tune_at = int(len(model.layers)*0.6)
    for layer in model.layers[:fine_tune_at]:
        layer.trainable = False
    
    return model
