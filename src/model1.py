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

NUM_CLASSES = 2
NUM_FEATURES = 2
RANDOM_SEED = 42

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

    ss = StandardScaler()

    X_test2_trans = ss.fit_transform(X_test2)
    X_pred2 = clf2.predict(X_test2_trans)



    score = dict(
        mae=mean_absolute_error(y_test2, X_pred2),
        rmse=mean_squared_error(y_test2, X_pred2)
    )

    path_model = "model1.clf"
    # jb.dump(clf3, path_model)


    signature = infer_signature(X_test2_trans, clf2.predict(X_test2_trans))
    mlflow.sklearn.log_model(
        sk_model=clf2,
        artifact_path=path_model,
        registered_model_name="KNeighbors_model1",
        signature=signature
    )

    # with open("model2.json", "w") as score_file:
    #     json.dump(score, score_file, indent=4)

