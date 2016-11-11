from os import listdir
import numpy as np
from PIL import Image
from sklearn.svm import SVC
from sklearn import decomposition, preprocessing
import sys
from sklearn.metrics import precision_score, recall_score


class ImageItem(object):
    def __init__(self):
        pass


def pca_transform(data_1, data_2, n_components):
    pca = decomposition.PCA(n_components=n_components)
    data = np.concatenate((data_1, data_2))
    pca.fit(data)
    new_data = pca.transform(data)
    return new_data[:data_1.shape[0], :], new_data[data_1.shape[0]:, :]


def method_1(im):
    # rgb to grey
    new_im = im.convert("L")

    # downsize
    new_size = (int(new_im.width / 4), int(new_im.height / 4))
    new_im = new_im.resize(new_size)

    image = np.array(new_im).reshape(new_im.width * new_im.height)
    return image


def readfiles(path, img_preprocess):
    all_files_name = listdir(path)
    images = [0] * len(all_files_name)
    labels = [0] * len(all_files_name)

    for i, f_name in enumerate(all_files_name):
        im = Image.open(path + f_name)
        images[i] = img_preprocess(im)
        labels[i] = ord(f_name[0]) - ord('a')

    return np.array(images, dtype=np.float), np.array(labels, dtype=np.float)


def main():
    # Setting
    num_class = 26
    training_directory = sys.argv[1]
    test_directory = sys.argv[2]

    # Read, Preprocess and Downsize Image
    images, labels = readfiles(training_directory, img_preprocess=method_1)
    test_images, test_labels = readfiles(test_directory, img_preprocess=method_1)

    # Standardization
    images = preprocessing.scale(images)
    test_images = preprocessing.scale(test_images)

    # Feature Selection
    images, test_images = pca_transform(images, test_images, 6)

    # Training
    tmp_label = [0] * len(labels)
    model = [SVC() for i in range(num_class)]
    for i in range(num_class):
        for j in range(len(labels)):
            tmp_label[j] = 1 if i == labels[j] else 0
        model[i].fit(images, tmp_label)

    # Testing
    tmp_label = [0] * len(test_labels)
    print("label  precision   recall")
    for i in range(num_class):
        for j in range(len(test_labels)):
            tmp_label[j] = 1 if i == test_labels[j] else 0
        test_pred = model[i].predict(test_images)
        p = precision_score(tmp_label, test_pred)
        r = recall_score(tmp_label, test_pred)
        print("    %c     %.4f   %.4f" % (chr(i + 65), p, r))


if __name__ == "__main__":
    main()
