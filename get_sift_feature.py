import cv2 as cv
import os
import numpy as np


def calc_sift_feature(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sift = cv.xfeatures2d.SIFT_create(200)
    kps, des = sift.detectAndCompute(gray, None)

    return des
    
def get_sift_feature(path):
    path = os.path.normpath(path)
    flowers = {}

    if not os.path.exists(path):
        raise IOError('The directory "{}" is not exsits'.format(path))

    for root, dirs, files in os.walk(path):
        feature_set = np.float32([]).reshape(0, 128)
        for file in files:
            if file.endswith('.jpg'):
                file_path = os.path.join(root, file)
                img = cv.imread(file_path)
                des = calc_sift_feature(img)
                if des is not None:
                    feature_set = np.append(feature_set, des, axis=0)

        feature_counts = feature_set.shape[0]
        catogray = root.split(os.sep)[-1]
        if catogray != 'train':
            file_name = os.path.join(path, catogray)
            np.save(file_name, feature_set)
    
    print('Done!')


if __name__ == '__main__':
    get_sift_feature('flowers')