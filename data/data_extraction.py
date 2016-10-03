import os
import numpy as np
from scipy.io import savemat
from scipy.misc import imresize
from keras.utils import np_utils
from scipy.ndimage import imread


def get_labels(directory):

    label_counter = 0
    labels = {}
    for label in sorted(os.listdir(directory)):
        labels[label] = label_counter
        label_counter += 1

    return labels


def get_examples(directory):

    examples = 0
    class_directories = os.listdir(directory)
    for class_ in class_directories:
        class_directory = os.path.join(directory, class_)
        examples += len(os.listdir(class_directory))

    return examples


def get_data(directory):

    # get a dictionary of labels and the number of classes
    labels = get_labels(directory)

    # get the number of examples in the directory
    examples = get_examples(directory)

    # create empty data and label arrays
    x = np.empty((examples, 3, 224, 224), dtype='float32')
    y = np.empty((examples,), dtype='int16')

    # fill the arrays by looping over class directories
    counter = 0
    class_directories = os.listdir(directory)
    for class_ in class_directories:
        class_directory = os.path.join(directory, class_)
        class_images = os.listdir(class_directory)
        for image in class_images:
            image_path = os.path.join(class_directory, image)
            img = imread(image_path)
            img_resized = imresize(img, size=(224, 224, 3))
            x[counter] = np.transpose(img_resized, axes=(2, 0, 1))
            y[counter] = labels[class_]
            counter += 1
    assert counter == examples

    # convert integer label to one-hot binary vector
    y = np_utils.to_categorical(np.ndarray.astype(y, dtype='int16'), nb_classes=len(labels))

    return x, y


# define train and validation directories
train_directory = 'TRAIN_DIR'
valid_directory = 'VALIDATION_DIR'

# get training and validation data
x_train, y_train = get_data(train_directory)
x_valid, y_valid = get_data(valid_directory)

# save the data
data = {
    'x_train': x_train,
    'y_train': y_train,
    'x_valid': x_valid,
    'y_valid': y_valid
}
savemat(file_name='/home/dan/Halloween/data/halloween_data.mat', mdict=data)
