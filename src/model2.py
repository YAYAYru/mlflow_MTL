from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

import joblib as jb
import json

import mlflow.sklearn
from mlflow.models.signature import infer_signature

NUM_CLASSES = 3
NUM_FEATURES = 2
RANDOM_SEED = 42

mlflow.set_experiment("model2")
with mlflow.start_run():
    mlflow.sklearn.autolog()
    X_blob3, y_blob3 = make_blobs(n_samples=100,
        n_features=NUM_FEATURES, 
        centers=NUM_CLASSES, 
        cluster_std=1.5,
        random_state=RANDOM_SEED
    )


    X, y = X_blob3, y_blob3
    X_train3, X_test3, y_train3, y_test3 = train_test_split(
        X, y, test_size=0.4, random_state=42
    )



    clf3 = KNeighborsClassifier(3)
    clf3 = make_pipeline(StandardScaler(), clf3)
    clf3.fit(X_train3, y_train3)

    ss = StandardScaler()

    X_test3_trans = ss.fit_transform(X_test3)
    X_pred3 = clf3.predict(X_test3_trans)



    score = dict(
        mae=mean_absolute_error(y_test3, X_pred3),
        rmse=mean_squared_error(y_test3, X_pred3)
    )

    path_model = "model2.clf"
    # jb.dump(clf3, path_model)


    signature = infer_signature(X_test3_trans, clf3.predict(X_test3_trans))
    mlflow.sklearn.log_model(
        sk_model=clf3,
        artifact_path=path_model,
        registered_model_name="KNeighbors_model2",
        signature=signature
    )

    # with open("model2.json", "w") as score_file:
    #     json.dump(score, score_file, indent=4)

