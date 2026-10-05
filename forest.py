# Cross validation Balance Accuracy = 92.08 %
import sys
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import balanced_accuracy_score

TRAIN = "dry_bean_train.csv"
TEST = "dry_bean_test.csv"
N_TREES = 100
N_FOLDS = 5
MAX_DEPTH = None
MIN_LEAF = 2
rng = np.random.default_rng(0)

train = pd.read_csv(TRAIN)
test = pd.read_csv(TEST)

# the label column is the one that holds text (the variety names)
label_col = [c for c in train.columns if not pd.api.types.is_numeric_dtype(train[c])][0]
features = [c for c in train.columns if c != label_col]
X = train[features].to_numpy()
y = train[label_col].to_numpy()
X_test = test[features].to_numpy()


def build_forest(X, y, n_trees=N_TREES):
    """Each tree gets a random sample of beans (with repeats)
    and a random group of features."""
    n_samples, n_feats = X.shape
    k = max(2, int(np.sqrt(n_feats)) + 1)  # features per tree
    forest = []
    for _ in range(n_trees):
        rows = rng.integers(0, n_samples, n_samples)
        cols = rng.choice(n_feats, k, replace=False)
        tree = DecisionTreeClassifier(max_depth=MAX_DEPTH,
                                      min_samples_leaf=MIN_LEAF,
                                      random_state=int(rng.integers(1e9)))
        tree.fit(X[rows][:, cols], y[rows])
        forest.append((tree, cols))
    return forest


def predict_forest(forest, X):
    """Every tree votes, the most common answer wins."""
    votes = np.array([t.predict(X[:, cols]) for t, cols in forest])  # trees x beans
    result = []
    for j in range(votes.shape[1]):
        names, counts = np.unique(votes[:, j], return_counts=True)
        result.append(names[np.argmax(counts)])
    return np.array(result)


def cross_validation(X, y, n_folds=N_FOLDS):
    """Our own k-fold: shuffle, cut into piles, hide one pile at a time."""
    idx = rng.permutation(len(y))
    piles = np.array_split(idx, n_folds)
    scores = []
    for i in range(n_folds):
        test_idx = piles[i]
        train_idx = np.concatenate([piles[j] for j in range(n_folds) if j != i])
        forest = build_forest(X[train_idx], y[train_idx])
        pred = predict_forest(forest, X[test_idx])
        scores.append(balanced_accuracy_score(y[test_idx], pred))
    return np.mean(scores) * 100


cv_score = cross_validation(X, y)
print(f"Cross validation Balance Accuracy = {cv_score:.2f} %")

# train on ALL training beans, predict the test beans
final = build_forest(X, y)
out = test.copy()
out["Target"] = predict_forest(final, X_test)
out.to_csv("forest.csv", index=False)
print("Saved forest.csv")

# write the score into the first line of this file
path = sys.argv[0]
with open(path) as f:
    lines = f.read().split("\n")
lines[0] = f"# Cross validation Balance Accuracy = {cv_score:.2f} %"
with open(path, "w") as f:
    f.write("\n".join(lines))
