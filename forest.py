# Cross validation Balance Accuracy = 93.63 %
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score

SEED = 42


def load_data():
    train = pd.read_csv("dry_bean_train.csv")
    test = pd.read_csv("dry_bean_test.csv")
    X = train.drop(columns="Class").to_numpy()
    y = train["Class"].to_numpy()
    return X, y, test


def make_folds(y, k=5, seed=SEED):
    """Stratified k-fold, coded by hand: shuffle each class's indices and deal
    them round-robin into k folds so every fold has the same class mix."""
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        for i, sample in enumerate(idx):
            folds[i % k].append(sample)
    return [np.array(f) for f in folds]


def cross_validate(make_model, X, y, k=5, seed=SEED):
    """Train on k-1 folds, score on the held-out fold, repeat k times.
    make_model is a function returning a fresh, untrained model."""
    folds = make_folds(y, k, seed)
    scores = {"balanced_acc": [], "acc": [], "macro_f1": []}
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        model = make_model()
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[val_idx])
        scores["balanced_acc"].append(balanced_accuracy_score(y[val_idx], pred))
        scores["acc"].append(accuracy_score(y[val_idx], pred))
        scores["macro_f1"].append(f1_score(y[val_idx], pred, average="macro"))
    return {name: (np.mean(v), np.std(v)) for name, v in scores.items()}


def report(name, res):
    ba, ba_sd = res["balanced_acc"]
    print(f"{name:40s} BalAcc {ba*100:6.2f} ±{ba_sd*100:4.2f}   "
          f"Acc {res['acc'][0]*100:6.2f}   MacroF1 {res['macro_f1'][0]*100:6.2f}")


def explore_trees(X, y):
    print("--- Single decision trees ---")
    for criterion in ["gini", "entropy"]:
        for max_depth in [None, 5, 8, 12]:
            for min_leaf in [1, 5, 20]:
                res = cross_validate(lambda: DecisionTreeClassifier(
                    criterion=criterion, max_depth=max_depth,
                    min_samples_leaf=min_leaf, random_state=SEED), X, y)
                report(f"{criterion} depth={max_depth} min_leaf={min_leaf}", res)


class Forest:
    """Hand-built random forest. Each tree is trained on:
      - a bootstrap sample of the rows (drawn with replacement), and
      - a random subset of the features (feature_frac of all columns).
    Prediction is a simple majority vote between the trees."""

    def __init__(self, n_trees=100, feature_frac=0.6, max_depth=None,
                 min_samples_leaf=1, split_features=None, seed=SEED):
        self.n_trees = n_trees
        self.feature_frac = feature_frac
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.split_features = split_features  # extra per-split randomness (tree option)
        self.seed = seed

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        n_samples, n_features = X.shape
        n_keep = max(1, int(round(self.feature_frac * n_features)))
        self.classes_ = np.unique(y)
        self.trees = []  # list of (tree, feature indices it was trained on)
        for _ in range(self.n_trees):
            rows = rng.integers(0, n_samples, n_samples)           # bootstrap
            cols = np.sort(rng.choice(n_features, n_keep, replace=False))
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                max_features=self.split_features,
                random_state=int(rng.integers(1_000_000_000)))
            tree.fit(X[rows][:, cols], y[rows])
            self.trees.append((tree, cols))
        return self

    def predict(self, X):
        # votes[i, c] = how many trees voted class c for sample i
        votes = np.zeros((X.shape[0], len(self.classes_)), dtype=int)
        class_pos = {c: i for i, c in enumerate(self.classes_)}
        for tree, cols in self.trees:
            pred = tree.predict(X[:, cols])
            votes[np.arange(len(pred)), [class_pos[p] for p in pred]] += 1
        return self.classes_[votes.argmax(axis=1)]


def explore_forests(X, y):
    print("--- Forests ---")
    # 1) How many trees? (more trees -> less variance, with diminishing returns)
    for n_trees in [10, 50, 100, 200]:
        res = cross_validate(lambda: Forest(n_trees=n_trees), X, y)
        report(f"n_trees={n_trees}", res)
    # 2) How many features per tree, and per-split randomness
    for frac in [0.3, 0.5, 0.7, 1.0]:
        for split in [None, "sqrt"]:
            res = cross_validate(lambda: Forest(n_trees=100, feature_frac=frac,
                                                split_features=split), X, y)
            report(f"feature_frac={frac} split_features={split}", res)
    # 3) Tree size inside the forest
    for max_depth, min_leaf in [(None, 1), (None, 3), (None, 5), (12, 1), (12, 3)]:
        res = cross_validate(lambda: Forest(n_trees=100, max_depth=max_depth,
                                            min_samples_leaf=min_leaf), X, y)
        report(f"max_depth={max_depth} min_leaf={min_leaf}", res)


EXPLORE = False  # set True to rerun the tree/forest hyperparameter searches (~25 min)

# Best settings found by explore_forests: 100 trees, 60% of features per tree,
# fully grown trees. More trees (200) or limiting depth did not help.
def best_forest():
    return Forest(n_trees=100, feature_frac=0.6, max_depth=None, min_samples_leaf=1)


if __name__ == "__main__":
    X, y, test = load_data()
    if EXPLORE:
        explore_trees(X, y)
        explore_forests(X, y)

    report("Final forest (5-fold CV)", cross_validate(best_forest, X, y))

    # Train on all labelled data, then predict the unlabelled test set
    model = best_forest().fit(X, y)
    out = test.copy()
    out["Target"] = model.predict(test.to_numpy())
    out.to_csv("forest.csv", index=False)
    print("Wrote forest.csv:", out["Target"].value_counts().to_dict())
