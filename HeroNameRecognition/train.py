from model import _base_network
from utils import _load_pickle
import cv2
import numpy as np
from config import config
import tensorflow as tf
import tensorflow_addons as tfa

def load_data(img_pkl, label_pkl):
    faces = _load_pickle(img_pkl)
    faceResizes = []
    for face in faces:
        face_rz = cv2.resize(face, (224, 224))
        faceResizes.append(face_rz)

    X_train = np.stack(faceResizes)
    y_train = _load_pickle(label_pkl)

    return X_train, y_train

def train(config_model):
    X_train, y_train = load_data(config_model.train_img_pkl, config_model.train_label_pkl)
    X_val, y_val = load_data(config_model.val_img_pkl, config_model.val_label_pkl)
    
    model = _base_network()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tfa.losses.TripletSemiHardLoss())

    gen_train = tf.data.Dataset.from_tensor_slices((X_train, y_train)).repeat().shuffle(1024).batch(config_model.batch_size)
    gen_val = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(config_model.batch_size)
    history = model.fit(gen_train,
                        steps_per_epoch = int(X_train.shape[0] / config_model.batch_size),
                        epochs=config_model.num_epoch,
                        validation_data=gen_val)
    model.save("model/model_triplot.h5")

if __name__ == "__main__":
    config_model = config()
    train(config_model)