# Hàm load model
## Load model từ Caffe
import cv2
import os
import numpy as np
from imutils import paths
import pickle
import sys

def _save_pickle(obj, file_path):
  with open(file_path, 'wb') as f:
    pickle.dump(obj, f)

def _load_torch(model_path_fl):
  """
  model_path_fl: Link file chứa weigth của model
  """
  model = cv2.dnn.readNetFromTorch(model_path_fl)
  return model

def _image_read(image_path):
  """
  input:
    image_path: link file ảnh
  return:
    image: numpy array của ảnh
  """
  image = cv2.imread(image_path)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  return image

def _model_processing(dataset_path):
  """
  face_scale_thres: Ngưỡng (W, H) để chấp nhận một khuôn mặt.
  """
  image_links = list(paths.list_images(dataset_path))
  images_file = [] 
  y_labels = []
  faces = []
  total = 0
  for image_link in image_links:
    split_img_links = image_link.split("\\") #on colab, replace with: split_img_links = image_link.split("/")
    # Lấy nhãn của ảnh
    name = split_img_links[-2]
    # name = split_img_links[-1][:-4]
    # Đọc ảnh
    image = _image_read(image_link)
    (h, w) = image.shape[:2]
    # Detect vị trí các khuôn mặt trên ảnh. Gỉa định rằng mỗi bức ảnh chỉ có duy nhất 1 khuôn mặt của chủ nhân classes.
    if True:
      # Lấy ra face
      face = image
      if face is not None:
        faces.append(face)
        y_labels.append(name)
        images_file.append(image_links)
        total += 1
      else:
        next
  return faces, y_labels, images_file

def create_pkl_data(in_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    dataset_path = os.path.join(in_dir)

    faces, y_labels, images_file = _model_processing(dataset_path)

    _save_pickle(faces, os.path.join(out_dir, "./faces.pkl"))
    _save_pickle(y_labels, os.path.join(out_dir, "./y_labels.pkl"))
    _save_pickle(images_file, os.path.join(out_dir, "./images_file.pkl"))

def _load_pickle(file_path):
    with open(file_path, 'rb') as f:
      obj = pickle.load(f)
    return obj