import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt 

PATH_FOLDER = "../data/raw/"

PATH_NPY_X_TRAIN = PATH_FOLDER + "x_train.npy"
PATH_NPY_X_TEST = PATH_FOLDER + "x_test.npy"
PATH_NPY_Y_TRAIN = PATH_FOLDER + "y_train.npy"
PATH_NPY_Y_TEST = PATH_FOLDER + "y_test.npy"

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

np.save(PATH_NPY_X_TRAIN, x_train)
np.save(PATH_NPY_X_TEST, x_test)
np.save(PATH_NPY_Y_TRAIN, y_train)
np.save(PATH_NPY_Y_TEST, y_test)

print('Shape of CIFAR10 dataset: \n', )
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# Get number of instances for each class

airplane_count = 0
automobile_count = 0
bird_count = 0
cat_count = 0
deer_count = 0
dog_count = 0
frog_count = 0
horse_count = 0
ship_count = 0
truck_count = 0

for sample in y_train:
    if sample == 0:
        airplane_count += 1
    elif sample == 1:
        automobile_count += 1
    elif sample == 2:
        bird_count += 1
    elif sample == 3:
        cat_count += 1
    elif sample == 4:
        deer_count += 1
    elif sample == 5:
        dog_count += 1
    elif sample == 6:
        frog_count += 1
    elif sample == 7:
        horse_count += 1
    elif sample == 8:
        ship_count += 1
    elif sample == 9:
        truck_count += 1
        
print('\n')
print(f'Number of samples of the class airplane: {airplane_count}')
print(f'Number of samples of the class automobile: {automobile_count}')
print(f'Number of samples of the class bird: {bird_count}')
print(f'Number of samples of the class cat: {cat_count}')
print(f'Number of samples of the class deer: {deer_count}')
print(f'Number of samples of the class dog: {dog_count}')
print(f'Number of samples of the class frog: {frog_count}')
print(f'Number of samples of the class horse: {horse_count}')
print(f'Number of samples of the class ship: {ship_count}')
print(f'Number of samples of the class truck: {truck_count}')

print('\nThe dataset is completely balanced')

