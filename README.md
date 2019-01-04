# flower-recognition

data resource comes from kaggle datasets 'flower recognition'

# data spliting

file: split_data.py

split datasets into training set and testing(validation) set for training model

# feature extraction

file: get_sift_feature.py vovabulary.py

extract image's sift feature for classifing different flowers, and get 'vocabulary' of every kind of flower by kmeans cluster algrithom

# training model

file: model.py

get feature vector(feature hist) of all flowers as the training data of model to train model, the accuarcy is 94.2%

# model validation

file: model_test.py

accuracy: 96.0%

# testing model

file: classify.py

download some pictures from website and test the model, and show the results with python lib matplotlib.pyplot
