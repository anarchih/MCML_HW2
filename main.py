from os import listdir
import numpy as np
from PIL import Image
from sklearn.svm import SVC
# from sklearn.metrics import precision_score, recall_score

class ImageItem(object):
    def __init__(self):
        pass


def test(im):
    # rgb to grey
    new_im = im.convert("L")

    # downsize
    new_size = (int(new_im.width / 4), int(new_im.height / 4))
    new_im = new_im.resize(new_size)

    image = np.array(new_im).reshape(new_im.width * new_im.height)
    return image


def readfiles(path, img_preprocess):
    all_files_name = listdir(path)[:1000]
    images = [0] * len(all_files_name)
    labels = [0] * len(all_files_name)

    for i, f_name in enumerate(all_files_name):
        im = Image.open(path + f_name)
        images[i] = img_preprocess(im)
        labels[i] = ord(f_name[0]) - ord('a')

    return images, labels


def main():
    # Setting
    num_class = 26


    # Read, Preprocess and Downsize Image
    images, labels = readfiles("training/", img_preprocess=test)
    test_images, test_labels = readfiles("test/", img_preprocess=test)

    # Feature Selection


    # Training
    tmp_label = [0] * len(labels)
    model = [SVC() for i in range(num_class - 1)]
    for i in range(num_class - 1):
        for j in range(len(labels)):
            tmp_label[j] = 1 if i == labels[j] else 0
        model[i].fit(images, tmp_label)

    # Testing



if __name__ == "__main__":
    main()
