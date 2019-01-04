import cv2 as cv
import os
import numpy as np


def get_file_path(file_path):
    flowers = {}
    file_path = os.path.normpath(file_path)
    if not os.path.exists(file_path):
        raise IOError('File path "{}" is not exists!'.format(file_path))
    for directory in os.listdir(file_path):
        if directory not in flowers:
            flowers[directory] = []
        subpath = os.path.join(file_path, directory)
        for file_name in os.listdir(subpath):
            file = os.path.join(subpath, file_name)
            if os.path.isfile(file) and file_name.endswith('.jpg'):
                flowers[directory].append(file)
    
    return flowers
    
def split_data(path):
    flowers = get_file_path(path)
    
    train_images, test_images = {}, {}
    # select 80 percentage of all data as training data
    for flower, file_list in flowers.items():
        n_flowers = len(file_list)
        n_train = int(0.8 * n_flowers)
        
        for i in range(n_train):
            if flower not in train_images:
                train_images[flower] = []
            train_images[flower].append(file_list[i])
        
        for i in range(n_train, n_flowers):
            if flower not in test_images:
                test_images[flower] = []
            test_images[flower].append(file_list[i])
            
    # save training data and testing data into 'train' directory and 'test' directory seperately
    train_data = 'datasets/flowers/train'
    if not os.path.exists(train_data):
        os.makedirs(train_data)
    for flower, file_list in train_images.items():
        flower_dir = os.path.join(train_data, flower)
        if not os.path.exists(flower_dir):
            os.makedirs(flower_dir)
        for image in file_list:
            img = cv.imread(image)
            image_name = image.split(os.sep)[-1]
            image_path = os.path.join(flower_dir, image_name)
            cv.imwrite(image_path, img)
            
    test_data = 'datasets/flowers/test'
    if not os.path.exists(test_data):
        os.makedirs(test_data)
    for flower, file_list in test_images.items():
        flower_dir = os.path.join(test_data, flower)
        if not os.path.exists(flower_dir):
            os.makedirs(flower_dir)
        for image in file_list:
            img = cv.imread(image)
            image_name = image.split(os.sep)[-1]
            image_path = os.path.join(flower_dir, image_name)
            cv.imwrite(image_path, img)
            
    print('Done!')


if __name__ == '__main__':
    split_data('flowers')