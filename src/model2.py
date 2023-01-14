import tensorflow as tf
import time
import numpy as np

PATH_MODEL = "../models/model2.h5"

PATH_FOLDER = "../data/processed/"
PATH_NPY_X_TRAIN = PATH_FOLDER + "x_train.npy"
PATH_NPY_X_TEST = PATH_FOLDER + "x_test.npy"
PATH_NPY_Y_TRAIN = PATH_FOLDER + "y_train_2.npy"
PATH_NPY_Y_TEST = PATH_FOLDER + "y_test_2.npy"

x_train = np.load(PATH_NPY_X_TRAIN)
x_test = np.load(PATH_NPY_X_TEST)
y_train = np.load(PATH_NPY_Y_TRAIN)
y_test = np.load(PATH_NPY_Y_TEST)

print("x_train.shape", x_train.shape)
print("x_test.shape", x_test.shape)
print("y_train.shape", y_train.shape)
print("y_test.shape", y_test.shape)

# BUILD MODEL

def create_task_learning_model():

    inputs = tf.keras.layers.Input(shape=(32, 32, 3), name='input')

    main_branch = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=1)(inputs)
    main_branch = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(main_branch)
    main_branch = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1)(main_branch)
    main_branch = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(main_branch)
    main_branch = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=1)(main_branch)
    main_branch = tf.keras.layers.Flatten()(main_branch)
    main_branch = tf.keras.layers.Dense(3512, activation='relu')(main_branch)

    task_2_branch = tf.keras.layers.Dense(512, activation='relu')(main_branch)
    task_2_branch = tf.keras.layers.Dense(256, activation='relu')(task_2_branch)
    task_2_branch = tf.keras.layers.Dense(100, activation='relu')(task_2_branch)
    task_2_branch = tf.keras.layers.Dense(2, activation='sigmoid')(task_2_branch)

    model = tf.keras.Model(inputs = inputs, outputs = [task_2_branch])
    model.summary()
    return model



def compile_task_model(model):
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
        
    return model


# FIT BATCH OF MODELS

def fit_batch():

    print('Starting training on batch of models for multitasks ', '\n\n')
    
    model2 = create_task_learning_model()
    model2 = compile_task_model(model2)

    start = time.time()
    model2_history = model2.fit(x_train, y_train,
                        epochs=15, batch_size=128, verbose=0)

    print(f'Training time: {time.time() - start}\n')
    return model2_history, model2
        

training_history, trained_model = fit_batch()
trained_model.save(PATH_MODEL)

new_model = tf.keras.models.load_model(PATH_MODEL)
new_model.summary()

print('Task2 evaluate: ', trained_model.evaluate(x_test, y_test))