from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score

import joblib as jb
import json

import mlflow
from mlflow.models.signature import infer_signature
from mlflow.pyfunc import PythonModel

NUM_CLASSES = 3
NUM_FEATURES = 2
RANDOM_SEED = 0


class CustomModel1(mlflow.pyfunc.PythonModel):
    def __init__(self, clf):
        self.clf = clf
        X, y = make_blobs(n_samples=100,
            n_features=NUM_FEATURES, 
            centers=NUM_CLASSES, 
            cluster_std=1.5,
            random_state=RANDOM_SEED
        )
        self.ss = StandardScaler()
        self.ss.fit(X)


    def predict(self, context, model_input):
        # pre_data = self.preprocessing(model_input)
        return self.clf.predict(model_input)


    def preprocessing(self, model_input):
        return self.ss.transform(model_input)


mlflow.set_experiment("model1")
with mlflow.start_run():
    mlflow.sklearn.autolog()
    X_blob2, y_blob2 = make_blobs(n_samples=100,
        n_features=NUM_FEATURES, 
        centers=NUM_CLASSES, 
        cluster_std=1.5,
        random_state=RANDOM_SEED
    )


    X, y = X_blob2, y_blob2
    X_train2, X_test2, y_train2, y_test2 = train_test_split(
        X, y, test_size=0.4, random_state=42
    )

    clf2 = KNeighborsClassifier(3)
    clf2 = make_pipeline(StandardScaler(), clf2)
    clf2.fit(X_train2, y_train2)

    path_custom_model = "model1"
    cm = CustomModel1(clf2)

    signature = infer_signature(X_test2, clf2.predict(X_test2))
    model_info = mlflow.pyfunc.log_model(
        python_model=cm,
        artifact_path=path_custom_model,
        registered_model_name="Pre_KNeighbors_model1",
        signature=signature
    )

    loaded_model = mlflow.pyfunc.load_model(model_uri=model_info.model_uri)
    y_pred2 = loaded_model.predict(X_test2)
    score = dict(
        mae=mean_absolute_error(y_test2, y_pred2),
        rmse=mean_squared_error(y_test2, y_pred2),
        acc=accuracy_score(y_test2, y_pred2)
    )
    print("score", score)
