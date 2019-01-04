import pickle
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from get_sift_feature import calc_sift_feature


def get_sift_vector(img, directory):
    vocabulary_path = 'datasets/flowers/vocabulary'
    if not os.path.exists(vocabulary_path):
        raise IOError('The directory "{}" is not exist'.format(vocabulary_path))
    
    feature_vector = np.zeros((1, 50))
    labels, centers = np.load(os.path.join(vocabulary_path, directory + '_vocabulary.npy'))
    
    feature = calc_sift_feature(img)
    if feature is not None:
        for i in range(feature.shape[0]):
            f = feature[i]
            diff_feature = np.tile(f, (50, 1)) - centers
            distance_feature = (diff_feature ** 2).sum(axis=1)
            sorted_index = distance_feature.argsort()
            index = sorted_index[0]
            feature_vector[0][index] += 1

    return feature_vector

def train_model():
    train_path = 'datasets/flowers/train'
    train = os.listdir(train_path)
    label_code = {'daisy': [[1]], 'dandelion': [[2]], 'rose': [[3]], 'sunflower': [[4]], 'tulip': [[5]]}
    feature_vectors = np.float32([]).reshape(0, 50)
    labels = np.int32([]).reshape(0, 1)

    for directory in train:
        dir_path = os.path.join(train_path, directory)
        if os.path.isdir(dir_path):
            image_names = os.listdir(dir_path)
            for image_name in image_names:
                if image_name.endswith('jpg'):
                    image_path = os.path.join(dir_path, image_name)
                    img = cv.imread(image_path)
                    feature_vector = get_sift_vector(img, directory)
                    feature_vectors = np.append(feature_vectors, feature_vector, axis=0)
                    labels = np.append(labels, label_code[directory], axis=0)

    param_grid = {'gamma': [0.001, 0.01, 0.1, 1, 10, 100], 'C': [0.001, 0.01, 0.1, 1, 10, 100]}
    model = SVC(random_state=7)
    grid_search = GridSearchCV(model, param_grid, cv=10)
    grid_search.fit(feature_vectors, labels.ravel())
    print('best params:', grid_search.best_params_)
    print('best score:', grid_search.best_score_)
    
    with open('datasets/flowers/model.pkl', 'wb') as f:
        pickle.dump(grid_search, f)

    return grid_search


if __name__ == '__main__':
    train_model()