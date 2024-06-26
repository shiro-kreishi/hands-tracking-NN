import cv2
import os
import numpy as np


def load_dataset(data_dir, label):
    images = []
    labels = []

    for file in os.listdir(data_dir):
        img_path = os.path.join(data_dir, file)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (64, 64))  # Изменим размер для согласованности
        images.append(img)
        labels.append(label)

    return np.array(images), np.array(labels)


