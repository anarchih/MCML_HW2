from os import listdir
import numpy as np
from PIL import Image
from sklearn.svm import SVC
from sklearn import decomposition, preprocessing
import sys
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import classification_report


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
    model_method = sys.argv[3]
    kernel_type = sys.argc[4]
    feature_scaling = sys.argc[5]

    # Read, Preprocess and Downsize Image
    images, labels = readfiles(training_directory, img_preprocess=method_1)
    test_images, test_labels = readfiles(test_directory, img_preprocess=method_1)

    # Feature scaling
    # Rescaling the range in [0, 1]
    if feature_scaling == 1:
        images = preprocessing.minmax_scale(images)
        test_images = preprocessing.minmax_scale(test_images)
    # Standardization
    if feature_scaling == 2:
        images = preprocessing.scale(images)
        test_images = preprocessing.scale(test_images)
    # Normalization (L2 norm)
    if feature_scaling == 3:
        images = preprocessing.normalize(images)
        test_images = preprocessing.normalize(test_images)

    # Feature Selection
    images, test_images = pca_transform(images, test_images, 3)

    # Training and Testing
    # create 26 two-class classifiers
    if model_method == 1:
        # Training
        tmp_label = [0] * len(labels)
        model = [SVC(kernel=kernel_type) for i in range(num_class)]
        for i in range(num_class):
            for j in range(len(labels)):
                tmp_label[j] = 1 if i == labels[j] else 0
            model[i].fit(images, tmp_label)

        # Testing
        tmp_label = [0] * len(test_labels)
        avg_p = 0
        avg_r = 0
        print("label  precision   recall")
        for i in range(num_class):
            for j in range(len(test_labels)):
                tmp_label[j] = 1 if i == test_labels[j] else 0
            test_pred = model[i].predict(test_images)
            p = precision_score(tmp_label, test_pred)
            r = recall_score(tmp_label, test_pred)
            avg_p += p
            avg_r += r
            print("    %c     %.4f   %.4f" % (chr(i + 65), p, r))
        avg_p /= 26
        avg_r /= 26
        print("-------------------------")
        print("  %s     %.4f   %.4f" % "AVG", avg_p, avg_r)

    # use the default model
    if model_method == 2:
        # Training
        model = SVC(kernel=kernel_type)
        model.fit(images, labels)

        # Testing
        test_pred = model.predict(test_images)
        names = [chr(i + 65) for i in range(num_class)]
        print(classification_report(test_labels, test_pred, target_names=names))


if __name__ == "__main__":
    main()
