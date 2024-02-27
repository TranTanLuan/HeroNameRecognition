from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from train import load_data
from config import config
import tensorflow as tf
import tensorflow_addons as tfa

def _most_similarity(embed_vecs, vec, labels):
    sim = cosine_similarity(embed_vecs, vec)
    sim = np.squeeze(sim, axis = 1)
    argmax = np.argsort(sim)[::-1][:1]
    label = [labels[idx] for idx in argmax][0]
    return label

def test(model, X_train, y_train, X_test, y_test):
    X_train_vec = model.predict(X_train)
    X_test_vec = model.predict(X_test)
    y_preds = []
    for vec in X_test_vec:
        vec = vec.reshape(1, -1)
        y_pred = _most_similarity(X_train_vec, vec, y_train)
        y_preds.append(y_pred)

    acc = (accuracy_score(y_preds, y_test))
    return acc

if __name__ == "__main__":
    config_model = config()
    X_train, y_train = load_data(config_model.train_img_pkl, config_model.train_label_pkl)
    X_test, y_test = load_data(config_model.test_img_pkl, config_model.test_label_pkl)
    model = tf.keras.models.load_model(config_model.pretrained_path, custom_objects={'loss': tfa.losses.TripletSemiHardLoss()})
    print(test(model, X_train, y_train, X_test, y_test))