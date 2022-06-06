import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras import regularizers
import os
import numpy as np
import config

class ArcFace(Layer):
    def __init__(self, n_classes=10, s=30.0, m=0.50, regularizer=None, **kwargs):
        super(ArcFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer = regularizers.get(regularizer)

    def build(self, input_shape):
        super(ArcFace, self).build(input_shape[0])
        self.W = self.add_weight(name='W',
                                shape=(input_shape[0][-1], self.n_classes),
                                initializer='glorot_uniform',
                                trainable=True,
                                regularizer=self.regularizer)

    def call(self, inputs):
        x, y = inputs
        c = K.shape(x)[-1]
        # normalize feature
        x = tf.nn.l2_normalize(x, axis=1)
        # normalize weights
        W = tf.nn.l2_normalize(self.W, axis=0)
        # dot product
        logits = x @ W
        # add margin
        # clip logits to prevent zero division when backward
        theta = tf.acos(K.clip(logits, -1.0 + K.epsilon(), 1.0 - K.epsilon()))
        target_logits = tf.cos(theta + self.m)
        # sin = tf.sqrt(1 - logits**2)
        # cos_m = tf.cos(logits)
        # sin_m = tf.sin(logits)
        # target_logits = logits * cos_m - sin * sin_m
        #
        logits = logits * (1 - y) + target_logits * y
        # feature re-scale
        logits *= self.s
        out = tf.nn.softmax(logits)

        return out

    def compute_output_shape(self, input_shape):
        return (None, self.n_classes)

def build_effarc_model():
    base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2M(include_top=False, weights='imagenet',
                                                                      input_shape=config.input_shape,)

    base_model.trainable = False

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
    x = tf.keras.layers.AveragePooling2D(pool_size=(15, 15))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(config.num_classes, activation=None,
                                    activity_regularizer=tf.keras.regularizers.L2(0.1))(x)
    output = ArcFace(num_classes=config.num_classes)([x, label])
    model = tf.keras.Model([inputs, label], output)
    
    return model
