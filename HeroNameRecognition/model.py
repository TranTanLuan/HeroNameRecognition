import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Lambda
from tensorflow.keras.models import Model

def _base_network():
  model = VGG16(include_top = False, weights = "imagenet", input_shape=(224, 224, 3))
  dense = Dense(128)(tf.keras.layers.Flatten()(model.layers[-1].output))
  norm2 = Lambda(lambda x: tf.math.l2_normalize(x, axis = 1))(dense)
  model = Model(inputs = [model.input], outputs = [norm2])
  return model