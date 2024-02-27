import os
import sys
import shutil
from tqdm import tqdm
import copy
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import cv2
from utils import create_pkl_data

def add_shadow_opencv(image, radius=3, offset=(5, 5)):
    shadow = cv2.blur(image, (radius, radius))

    offset_x, offset_y = offset
    image = cv2.addWeighted(shadow, 0.5, image, 0.5, 0)
    image[offset_y:, offset_x:] = image[offset_y:, offset_x:] * 0.5 + shadow[offset_y:, offset_x:] * 0.5
    return image

def light_white_rgb():
    r = np.random.randint(200, 255)
    g = np.random.randint(180, 255)
    b = np.random.randint(140, 255)
    return (b, g, r)

def draw_circle(image, thickness):
    radius = int(min(image.shape[0], image.shape[1]) / 2)
    center_coordinates = (radius, radius) 
    color = light_white_rgb() 
    image = cv2.circle(image, center_coordinates, radius, color, thickness)
    if np.random.choice([0, 1], p=[0.2, 0.8]):
        # Tạo bóng tối
        shadow = cv2.GaussianBlur(image, (3, 3), 0)
        # Thêm bóng tối lên trên đường viền
        image = cv2.addWeighted(image, 0.7, shadow, 0.3, 0)
        image = add_shadow_opencv(image)
    return image

def aug(image):
    IMG_SIZE = 224

    aug1 = tf.keras.Sequential([
        layers.Resizing(IMG_SIZE, IMG_SIZE),
        layers.RandomContrast(0.2),
        layers.RandomBrightness(factor=0.2),
    ])
    aug2 = tf.keras.Sequential([
        layers.RandomZoom(.5, .2, 'constant'),
        layers.RandomRotation(factor=(-0.05, 0.05), fill_mode='constant'),
    ])
    aug3 = tf.keras.Sequential([
        layers.RandomHeight(0.2),
        layers.RandomWidth(0.2),
    ])

    aug_type = np.random.choice([aug1, aug2, aug3], p=[0.8, 0.1, 0.1])
    thickness = 2
    image = draw_circle(image, thickness)
    image = aug_type(image)

    used_type = np.random.choice([0, 1, 2], p = [0.1, 0.4, 0.5])
    if used_type == 1:
        image = tf.image.adjust_saturation(image, 3)
        image = tf.image.adjust_brightness(image, 0.4)
    elif used_type == 2:
        image = tf.image.rgb_to_grayscale(image)

    image = image.numpy()
    return image

def create_data(in_dir, out_dir, num_imgs=50):
    os.makedirs(out_dir, exist_ok=True)
    list_characters = os.listdir(in_dir)

    for i in tqdm(range(len(list_characters))):
        out_dir_i = os.path.join(out_dir, list_characters[i])
        os.makedirs(out_dir_i, exist_ok=True)
        in_dir_i = os.path.join(in_dir, list_characters[i])
        list_names = os.listdir(in_dir_i)
        count = 0
        for idx in range(len(list_names)):
          shutil.copyfile(os.path.join(in_dir_i, list_names[idx]), os.path.join(out_dir_i, list_names[idx]))
          count += 1
        while count < num_imgs:
            name = np.random.choice(list_names)
            img = cv2.imread(os.path.join(in_dir_i, name))
            img = aug(img)
            cv2.imwrite(os.path.join(out_dir_i, f"{count}_"+ name), img)
            count += 1

if __name__ == "__main__":
    in_dir = sys.argv[1]
    out_dir = sys.argv[2]
    out_dir_pkl = sys.argv[3]
    num_per_type = int(sys.argv[4])
    create_data(in_dir, out_dir, num_per_type)
    create_pkl_data(out_dir, out_dir_pkl)