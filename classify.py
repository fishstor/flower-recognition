import matplotlib.pyplot as plt
import pickle


def classify(model):
    path = 'datasets/flowers'
    images = [image for image in os.listdir(path) if image.endswith('jpg')]
    label_code = {1: 'daisy', 2: 'dandelion', 3: 'rose', 4: 'sunflower', 5: 'tulip'}
    feature_vectors = np.float32([]).reshape(0, 50)
    
    for image in images:
        image_path = os.path.join(path, image)
        img = cv.imread(image_path)
        feature_vector = get_sift_vector(img, image.split('.')[0][:-1])
        feature_vectors = np.append(feature_vectors, feature_vector, axis=0)
        
    res = model.predict(feature_vectors)
    i = 1
    fig = plt.figure(figsize=(50, 30))
    
    for image in images:
        image_path = os.path.join(path, image)
        img = cv.imread(image_path)
        b, g, r = cv.split(img)
        img = cv.merge([r, g, b])
        
        plt.subplot(5, 3, i)
        image_name = image.split('.')[0][:-1]
        t = '{}({})'.format(image_name, label_code[res[i-1]])
        plt.title(t, fontsize=40)
        plt.imshow(img)
        plt.axis('off')
        i += 1
        
    plt.tight_layout()
    plt.show()


if __name__ == '__mian__':
    with open('datasets/flowers/model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    classify(model)