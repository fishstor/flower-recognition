import cv2 as cv
import os
import numpy as np


def test_model(svm):
    test_path = 'datasets/flowers/test'
    dirs = os.listdir(test_path)
    # print(dirs)

    label_code = {'daisy': [[1]], 'dandelion': [[2]], 'rose': [[3]], 'sunflower': [[4]], 'tulip': [[5]]}
    feature_vectors = np.float32([]).reshape(0, 50)
    labels = np.int32([]).reshape(0, 1)

    for directory in dirs:
        dir_path = os.path.join(test_path, directory)
        if os.path.isdir(dir_path):
            image_names = os.listdir(dir_path)
            for image_name in image_names:
                if image_name.endswith('jpg'):
                    image_path = os.path.join(dir_path, image_name)
                    img = cv.imread(image_path)
                    feature_vector = get_sift_vector(img, directory)
                    feature_vectors = np.append(feature_vectors, feature_vector, axis=0)
                    labels = np.append(labels, label_code[directory], axis=0)

    res = svm.predict(feature_vectors)
    res = res.reshape(-1, 1)

    n_correct = (labels == res).sum()
    accuracy = n_correct / len(res)
    print('accuracy:', accuracy)
    
