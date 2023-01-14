import tensorflow as tf
import numpy as np

PATH_FOLDER = "../data/processed/"
PATH_NPY_X_TEST = PATH_FOLDER + "x_test.npy"
PATH_NPY_Y_TEST_1 = PATH_FOLDER + "y_test_1.npy"
PATH_NPY_Y_TEST_2 = PATH_FOLDER + "y_test_2.npy"

PATH_MODEL_1 = "../models/model1.h5"
PATH_MODEL_2 = "../models/model2.h5"

x_test = np.load(PATH_NPY_X_TEST)
y_test_1 = np.load(PATH_NPY_Y_TEST_1)
y_test_2 = np.load(PATH_NPY_Y_TEST_2)

model1 = tf.keras.models.load_model(PATH_MODEL_1)
model1.summary()

model2 = tf.keras.models.load_model(PATH_MODEL_2)
model2.summary()

e1 = model1.evaluate(x_test, y_test_1)
print('Task1 evaluate: ', e1)

e2 = model2.evaluate(x_test, y_test_2)
print('Task2 evaluate: ', e2)

print('Multi task evaluate sum loss: ', e1[0] + e2[0])
print('Multi task evaluate  average accurate: ', (e1[1] + e2[1])/2)