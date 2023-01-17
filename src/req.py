import json
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()


print("x_train", x_train[:1].shape)
np_image1 = x_train[:1]

with open("sample.json", "w") as f:
    f.write(json.dumps(
        {
            "instances": np_image1.tolist()
        }
    ))

"""
https://santiagof.medium.com/effortless-models-deployment-with-mlflow-2b1b443ff157
!cat -A sample.json | curl http://0.0.0.0:8001/invocations \
                        --request POST \
                        --header 'Content-Type: application/json' \
                        --data-binary @-

"""