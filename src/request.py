import numpy as np
# import tensorflow as tf
import requests
import numpy as np
import random
import matplotlib.pyplot as plt

def generate_binary_labels(yy, animal_classes):
    yy_2 = np.array([[0] if y in animal_classes else [1] for y in yy])
    return yy_2

# (x_train, y_train_1), (x_test, y_test_1) = tf.keras.datasets.cifar10.load_data()

PATH_FOLDER = "../data/processed/"

PATH_NPY_X_TEST = PATH_FOLDER + "x_test.npy"
PATH_NPY_Y_TEST = PATH_FOLDER + "y_test.npy"

x_test = np.load(PATH_NPY_X_TEST)
y_test_1 = np.load(PATH_NPY_Y_TEST)

animal_classes = [2, 3, 4, 5, 6, 7]
y_test_2 = generate_binary_labels(y_test_1, animal_classes)

dict_binary_label = {
    0: "animal",
    1: "vehicle"
}
dict_multiclass_label = {
    0: "airplane", 
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck"
}


random_ind = random.randrange(0, x_test.shape[0], 1)
print(dict_binary_label[y_test_2[random_ind][0]], dict_multiclass_label[y_test_1[random_ind][0]])
# for n2, n1 in zip(y_test_2[5:20], y_test_1[5:20]):
#     print(dict_binary_label[n2], dict_multiclass_label[n1[0]])


# random_ind = random.randrange(0, x_test.shape[0], 1)
print("random_ind", random_ind)
image = x_test[random_ind,:,:,:]
# plt.imshow(image)
print("y_test_1", y_test_1.shape)
print(
    'Expected binary labels: ',
    dict_binary_label[y_test_2[random_ind][0]],
    '\nExpected multiclass labels: ',
    dict_multiclass_label[y_test_1[random_ind][0]]
)

np_image1 = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
j = {"instances": np_image1.tolist()}


r2 = requests.post('http://0.0.0.0:8002/invocations', json=j)
print(r2.status_code)
dict_r2 = r2.json()
print(
    "predicted binary labels:",
    dict_binary_label[np.argmax(dict_r2["predictions"][0])]
)


r1 = requests.post('http://0.0.0.0:8001/invocations', json=j)
print(r1.status_code)
dict_r1 = r1.json()
print(
    "predicted multiclass labels:",
    dict_multiclass_label[np.argmax(dict_r1["predictions"][0])],
)