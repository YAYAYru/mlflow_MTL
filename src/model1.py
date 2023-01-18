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

x_train = np.load(PATH_NPY_X_TRAIN)
x_test = np.load(PATH_NPY_X_TEST)
y_train = np.load(PATH_NPY_Y_TRAIN)
y_test = np.load(PATH_NPY_Y_TEST)

print("x_train.shape", x_train.shape)
print("x_test.shape", x_test.shape)
print("y_train.shape", y_train.shape)
print("y_test.shape", y_test.shape)


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


def create_task_learning_model(x_train_shape, y_train_shape):

    inputs = tf.keras.layers.Input(shape=(x_train_shape[1], x_train_shape[2], x_train_shape[3]), name='input')

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
    task_1_branch = tf.keras.layers.Dense( y_train_shape[1], activation='softmax')(task_1_branch)

    model = tf.keras.Model(inputs = inputs, outputs = [task_1_branch])
    model.summary()
    return model

experiment = mlflow.set_experiment("model1")

# Get Experiment Details
print("Experiment_id: {}".format(experiment.experiment_id))
print("Artifact Location: {}".format(experiment.artifact_location))
print("Tags: {}".format(experiment.tags))
print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))

with mlflow.start_run() as run:
    mlflow.tensorflow.autolog()

    print('Starting training on batch of models for multitasks ', '\n\n')
    
    model = create_task_learning_model(x_train.shape, y_train.shape)
    model.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

    start = time.time()
    model2_history = model.fit(x_train, y_train,
                        epochs=50, batch_size=128, verbose=0)

    print(f'Training time: {time.time() - start}\n')

    model.save(PATH_MODEL)

    new_model = tf.keras.models.load_model(PATH_MODEL)
    new_model.summary()

    e = model.evaluate(x_test, y_test)
    mlflow.log_metric("test_loss", e[0])
    print('Task1 evaluate test loss: ', e[0])
    mlflow.log_metric("test_acc", e[1])
    print('Task1 evaluate test acc: ', e[1])

    # model_uri = "runs:/{}/model1".format(run.info.run_id)
    # mv = mlflow.register_model(model_uri, "model1")
    # print("Name: {}".format(mv.name))
    # print("Version: {}".format(mv.version))

# mlflow models serve --no-conda -m file:///home/yayay/yayay/git/github/paper_mlflow/src/mlruns/137049049665508372/b78c51445cdb498aa89cfc4441d39eaf/artifacts/model -h 0.0.0.0 -p 8001

# export CUDA_VISIBLE_DEVICES='' Выключить GPU для предикта
# export CUDA_VISIBLE_DEVICES='0' Включить GPU для обучения
# https://datascience.stackexchange.com/questions/58845/how-to-disable-gpu-with-tensorflow