import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

PATH_FOLDER = "../data/raw/"
PATH_NPY_X_TRAIN = PATH_FOLDER + "x_train.npy"
PATH_NPY_X_TEST = PATH_FOLDER + "x_test.npy"
PATH_NPY_Y_TRAIN = PATH_FOLDER + "y_train.npy"
PATH_NPY_Y_TEST = PATH_FOLDER + "y_test.npy"
x_train = np.load(PATH_NPY_X_TRAIN)
x_test = np.load(PATH_NPY_X_TEST)
y_train = np.load(PATH_NPY_Y_TRAIN)
y_test = np.load(PATH_NPY_Y_TEST)

print("x_train.shape", x_train.shape)
print("x_test.shape", x_test.shape)
print("y_train.shape", y_train.shape)
print("y_test.shape", y_test.shape)

PATH_FOLDER = "../data/processed/"
PATH_NPY_X_TRAIN = PATH_FOLDER + "x_train.npy"
PATH_NPY_X_TEST = PATH_FOLDER + "x_test.npy"
PATH_NPY_Y_TRAIN_1 = PATH_FOLDER + "y_train_1.npy"
PATH_NPY_Y_TEST_1 = PATH_FOLDER + "y_test_1.npy"
PATH_NPY_Y_TRAIN_2 = PATH_FOLDER + "y_train_2.npy"
PATH_NPY_Y_TEST_2 = PATH_FOLDER + "y_test_2.npy"

def erase_classes(classes_to_drop, x_train, x_test, y_train, y_test):
    """
    :param classes_to_drop: list with the labels of the classes to be erased
    :return: datasets tuple without the classes: (
    """
    new_x_train, new_x_test, new_y_train,  new_y_test = list(), list(), list(), list()
    for train_sample, label in zip(x_train, y_train):
        if label not in classes_to_drop:
            new_x_train.append(train_sample)
            new_y_train.append(label)

    for test_sample, test_label in zip(x_test, y_test):
        if test_label not in classes_to_drop:
            new_x_test.append(test_sample)
            new_y_test.append(test_label)

    return np.array(new_x_train), np.array(new_x_test), np.array(new_y_train), np.array(new_y_test)


def generate_binary_labels(y_train, y_test, animal_classes):
    """
    :param y_train: training labels
    :param y_test: testing labels
    :param animal_classes: list of the labels which correspond to an animal
    :return: training and testing labels for binary classification, where
    animals' label is 0 and vehicle's label is 1
    """
    y_train_2 = [0 if y in animal_classes else 1 for y in y_train]
    y_test_2 = [0 if y in animal_classes else 1 for y in y_test]

    return y_train_2, y_test_2


def preprocess_data_cifar10(x_train, y_train_1, x_test, y_test_1):
    
    airplane_count, automobile_count, bird_count, cat_count, deer_count, dog_count, frog_count, horse_count, ship_count, truck_count = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    
    # 0 = animal, 1 = vehicle
    y_train_2, y_test_2 = generate_binary_labels(y_train_1, y_test_1, [2, 3, 4, 5, 6, 7])    
    
    # Print amount of instances of each class
    for label in y_train_1:
        if label == 0:
            airplane_count += 1
        elif label == 1:
            automobile_count += 1
        elif label == 2:
            bird_count += 1
        elif label == 3:
            cat_count += 1
        elif label == 4:
            deer_count += 1
        elif label == 5:
            dog_count += 1
        elif label == 6:
            frog_count += 1
        elif label == 7:
            horse_count += 1
        elif label == 8:
            ship_count += 1
        elif label == 9:
            truck_count += 1
        
    print(f'Number of samples of the class airplane: {airplane_count}')
    print(f'Number of samples of the class automobile: {automobile_count}')
    print(f'Number of samples of the class bird: {bird_count}')
    print(f'Number of samples of the class cat: {cat_count}')
    print(f'Number of samples of the class deer: {deer_count}')
    print(f'Number of samples of the class dog: {dog_count}')
    print(f'Number of samples of the class frog: {frog_count}')
    print(f'Number of samples of the class horse: {horse_count}')
    print(f'Number of samples of the class ship: {ship_count}')
    print(f'Number of samples of the class truck: {truck_count}\n')
    
    n_class_1 = 10
    n_class_2 = 2
    y_train_1 = to_categorical(y_train_1, n_class_1)
    y_test_1 = to_categorical(y_test_1, n_class_1)
    y_train_2 = to_categorical(y_train_2, n_class_2)
    y_test_2 = to_categorical(y_test_2, n_class_2)
    
    return x_train, y_train_1, y_train_2, x_test, y_test_1, y_test_2

x_train, y_train_1, y_train_2, x_test, y_test_1, y_test_2 = preprocess_data_cifar10(x_train, y_train, x_test, y_test)

print('\nShapeof modified CIFAR10 dataset: \n')
print(x_train.shape)
print(x_test.shape)
print(y_train_1.shape)
print(y_test_1.shape)
print(y_train_2.shape)
print(y_test_2.shape)

np.save(PATH_NPY_X_TRAIN, x_train)
np.save(PATH_NPY_X_TEST, x_test)
np.save(PATH_NPY_Y_TRAIN_1, y_train_1)
np.save(PATH_NPY_Y_TEST_1, y_test_1)
np.save(PATH_NPY_Y_TRAIN_2, y_train_2)
np.save(PATH_NPY_Y_TEST_2, y_test_2)
