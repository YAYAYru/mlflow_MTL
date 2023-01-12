from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline

import mlflow.sklearn

NUM_CLASSES = 3
NUM_FEATURES = 2
RANDOM_SEED = 42

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

mlflow.set_experiment("model2")
mlflow.sklearn.autolog()


clf3 = KNeighborsClassifier(3)
clf3 = make_pipeline(StandardScaler(), clf3)
clf3.fit(X_train3, y_train3)
