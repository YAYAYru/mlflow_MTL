from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline

import mlflow.sklearn


NUM_CLASSES = 2
NUM_FEATURES = 2
RANDOM_SEED = 42

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

mlflow.set_experiment("model1")
mlflow.sklearn.autolog()


clf2 = KNeighborsClassifier(3)
clf2 = make_pipeline(StandardScaler(), clf2)
clf2.fit(X_train2, y_train2)
print("clf2", clf2)
