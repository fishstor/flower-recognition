import cv2 as cv
import os
import numpy as np


def cluster():
    path = 'datasets/flowers/train'
    save_path = 'datasets/flowers/vocabulary'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    directory = os.listdir(path)
    for d in directory:
        file_path = os.path.join(path, d)
        if os.path.isfile(file_path):
            features = np.load(file_path)

            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 20, 0.1)
            flags = cv.KMEANS_RANDOM_CENTERS
            compactness, labels, centers = cv.kmeans(features, 50, None, criteria, 20, flags)

#             root = file_path.split('.')[0]
            save_name = os.path.join(save_path, d.split('.')[0] + '_vocabulary.npy')
            np.save(save_name, (labels, centers))
    
    print('Done!')
