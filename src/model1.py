import tensorflow as tf
import time
import numpy as np
import mlflow
import random
import os

PATH_MODEL = "../models/model1.h5"

PATH_FOLDER = "../data/processed/"
PATH_NPY_X_TRAIN = PATH_FOLDER + "x_train.npy"
PATH_NPY_X_TEST = PATH_FOLDER + "x_test.npy"
PATH_NPY_Y_TRAIN = PATH_FOLDER + "y_train_1.npy"
PATH_NPY_Y_TEST = PATH_FOLDER + "y_test_1.npy"

# https://stackoverflow.com/questions/36288235/how-to-get-stable-results-with-tensorflow-setting-random-seed
SEED = 0
def set_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
def set_global_determinism(seed=SEED):
    set_seeds(seed=seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
# Call the above function with seed value
set_global_determinism(seed=SEED)


experiment = mlflow.set_experiment("model1")

# Get Experiment Details
print("Experiment_id: {}".format(experiment.experiment_id))
print("Artifact Location: {}".format(experiment.artifact_location))
print("Tags: {}".format(experiment.tags))
print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))

with mlflow.start_run():
    mlflow.tensorflow.autolog()

    x_train = np.load(PATH_NPY_X_TRAIN)
    x_test = np.load(PATH_NPY_X_TEST)
    y_train = np.load(PATH_NPY_Y_TRAIN)
    y_test = np.load(PATH_NPY_Y_TEST)

    print("x_train.shape", x_train.shape)
    print("x_test.shape", x_test.shape)
    print("y_train.shape", y_train.shape)
    print("y_test.shape", y_test.shape)

    def create_task_learning_model():

        inputs = tf.keras.layers.Input(shape=(32, 32, 3), name='input')

        main_branch = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=1)(inputs)
        main_branch = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(main_branch)
        main_branch = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1)(main_branch)
        main_branch = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(main_branch)
        main_branch = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=1)(main_branch)
        main_branch = tf.keras.layers.Flatten()(main_branch)
        main_branch = tf.keras.layers.Dense(3512, activation='relu')(main_branch)

        task_1_branch = tf.keras.layers.Dense(1024, activation='relu')(main_branch)
        task_1_branch = tf.keras.layers.Dense(512, activation='relu')(task_1_branch)
        task_1_branch = tf.keras.layers.Dense(256, activation='relu')(task_1_branch)
        task_1_branch = tf.keras.layers.Dense(128, activation='relu')(task_1_branch)
        task_1_branch = tf.keras.layers.Dense(8, activation='softmax')(task_1_branch)

        model = tf.keras.Model(inputs = inputs, outputs = [task_1_branch])
        model.summary()
        return model


    def compile_task_model(model):
        model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
        return model


    def fit_batch():
        print('Starting training on batch of models for multitasks ', '\n\n')
        
        model = create_task_learning_model()
        model = compile_task_model(model)

        start = time.time()
        model2_history = model.fit(x_train, y_train,
                            epochs=15, batch_size=128, verbose=0)

        print(f'Training time: {time.time() - start}\n')
        return model2_history, model
            

    training_history, trained_model = fit_batch()
    trained_model.save(PATH_MODEL)

    new_model = tf.keras.models.load_model(PATH_MODEL)
    new_model.summary()

    e = trained_model.evaluate(x_test, y_test)
    mlflow.log_metric("test_loss", e[0])
    print('Task1 evaluate test loss: ', e[0])
    mlflow.log_metric("test_acc", e[1])
    print('Task1 evaluate test acc: ', e[1])
