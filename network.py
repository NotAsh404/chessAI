# Cross validation Balance Accuracy = 94.23 %
import copy
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import balanced_accuracy_score

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Data

def load_data():
    train = pd.read_csv("dry_bean_train.csv")
    test = pd.read_csv("dry_bean_test.csv")
    X = train.drop(columns="Class").to_numpy(dtype=np.float32)
    labels = train["Class"].to_numpy()
    classes = np.unique(labels) # sorted class names
    y = np.searchsorted(classes, labels) # name -> integer 0..6
    return X, y, classes, test


def make_folds(y, k=5, seed=SEED):
    """Stratified k-fold: shuffle each class's indices and deal them round-robin into k folds so every fold has the same class mix."""
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        for i, sample in enumerate(idx):
            folds[i % k].append(sample)
    return [np.array(f) for f in folds]


class Standardiser:
    """Scale each feature to mean 0, std 1. Fitted on training rows only so no information from validation/test rows leaks into training."""

    def fit(self, X):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + 1e-8
        return self

    def transform(self, X):
        return ((X - self.mean) / self.std).astype(np.float32)


class BeanDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = None if y is None else torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        if self.y is None:
            return self.X[i]
        return self.X[i], self.y[i]


# Model and loss

ACTIVATIONS = {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU, "tanh": nn.Tanh}


class BeanMLP(nn.Module):
    """
    Input -> [Linear -> BatchNorm -> activation -> Dropout] x n_hidden -> Linear
    """

    """
    The output layer gives one raw score (logit) per class.
    """

    def __init__(self, n_in, n_out, hidden=(64,), activation="relu", dropout=0.0):
        super().__init__()
        layers = []
        prev = n_in
        for width in hidden:
            layers += [nn.Linear(prev, width), nn.BatchNorm1d(width),
                       ACTIVATIONS[activation]()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = width
        self.hidden = nn.Sequential(*layers)
        self.out = nn.Linear(prev, n_out)

    def forward(self, x):
        return self.out(self.hidden(x))


def cross_entropy_loss(logits, targets):
    """
    Cross-entropy. Softmax turns logits into probabilities;
    the loss is -log(probability of the true class). Uses the log-sum-exp trick
    """

    """
    (subtract the row max) so exp() can't overflow. Summed over the batch and divided by the number of inputs, so the loss doesn't depend on batch size.
    """
    row_max = logits.max(dim=1, keepdim=True).values
    log_sum_exp = row_max.squeeze(1) + torch.log(torch.exp(logits - row_max).sum(dim=1))
    true_logit = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
    log_prob_true = true_logit - log_sum_exp
    return -log_prob_true.sum() / logits.shape[0]

# Training / evaluation loops
def train_one_epoch(model, loader, optimiser):
    model.train()                       # BatchNorm uses batch stats, Dropout on
    total_loss, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimiser.zero_grad()
        loss = cross_entropy_loss(model(xb), yb)
        loss.backward()
        optimiser.step()
        # loss is a per-sample mean; weight by batch size to accumulate exactly
        total_loss += loss.item() * len(xb)
        n += len(xb)
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader):
    model.eval()                        # BatchNorm uses running stats, Dropout off
    total_loss, n, preds, targets = 0.0, 0, [], []
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits = model(xb)
        total_loss += cross_entropy_loss(logits, yb).item() * len(xb)
        n += len(xb)
        preds.append(logits.argmax(dim=1).cpu())
        targets.append(yb.cpu())
    preds, targets = torch.cat(preds).numpy(), torch.cat(targets).numpy()
    return total_loss / n, balanced_accuracy_score(targets, preds)


@torch.no_grad()
def predict(model, X):
    model.eval()
    loader = DataLoader(BeanDataset(X), batch_size=1024)
    return torch.cat([model(xb.to(DEVICE)).argmax(dim=1).cpu() for xb in loader]).numpy()


BASELINE = dict(hidden=(64,), activation="relu", lr=1e-3, batch_size=64,
                epochs=50, dropout=0.0, weight_decay=0.0, patience=None)


def train_model(X_tr, y_tr, X_val, y_val, cfg, n_classes, log_dir=None):
    """
    Train one network. X_val is used for logging and (if patience is set) for early stopping: stop once validation loss hasn't improved for `patience` epochs, and restore the weights from the best epoch.
    """
    torch.manual_seed(SEED)
    model = BeanMLP(X_tr.shape[1], n_classes, cfg["hidden"], cfg["activation"],
                    cfg["dropout"]).to(DEVICE)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                                 weight_decay=cfg["weight_decay"])
    train_loader = DataLoader(BeanDataset(X_tr, y_tr), batch_size=cfg["batch_size"],
                              shuffle=True, drop_last=True)  # BatchNorm needs >1 sample
    val_loader = DataLoader(BeanDataset(X_val, y_val), batch_size=1024)
    writer = SummaryWriter(log_dir) if log_dir else None

    best_loss, best_state, best_epoch = float("inf"), None, 0
    for epoch in range(cfg["epochs"]):
        train_loss = train_one_epoch(model, train_loader, optimiser)
        val_loss, val_ba = evaluate(model, val_loader)
        if writer:
            writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
            writer.add_scalar("val_balanced_accuracy", val_ba, epoch)
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
        elif cfg["patience"] and epoch - best_epoch >= cfg["patience"]:
            break
    if writer:
        writer.close()
    if cfg["patience"]:
        model.load_state_dict(best_state)
    return model


# Cross validation

def cross_validate(cfg, X, y, n_classes, k=5, name=None):
    """
    k-fold CV. Inside each training fold, 10% is split off as an

    inner validation set for loss logging / early stopping, so the held-out

    fold is never seen during training.
    """
    folds = make_folds(y, k)
    scores = []
    for i in range(k):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        inner = make_folds(y[train_idx], 10, seed=SEED + i)
        val_idx = train_idx[inner[0]]
        fit_idx = train_idx[np.concatenate(inner[1:])]

        scaler = Standardiser().fit(X[fit_idx])
        model = train_model(scaler.transform(X[fit_idx]), y[fit_idx],
                            scaler.transform(X[val_idx]), y[val_idx], cfg, n_classes,
                            log_dir=f"runs/{name}/fold{i}" if name else None)
        pred = predict(model, scaler.transform(X[test_idx]))
        scores.append(balanced_accuracy_score(y[test_idx], pred))
    return np.mean(scores), np.std(scores)


def run(label, X, y, n_classes, **changes):
    cfg = {**BASELINE, **changes}
    mean, sd = cross_validate(cfg, X, y, n_classes, name=label)
    print(f"{label:28s} BalAcc {mean*100:6.2f} +-{sd*100:4.2f}", flush=True)
    return mean


def explore(X, y, n_classes):
    """
    Change one hyperparameter at a time relative to the baseline.
    """
    run("baseline", X, y, n_classes)
    for width in [16, 128, 256]:
        run(f"width_{width}", X, y, n_classes, hidden=(width,))
    for hidden in [(64, 64), (128, 64), (128, 128, 64)]:
        run("layers_" + "x".join(map(str, hidden)), X, y, n_classes, hidden=hidden)
    for act in ["leaky_relu", "tanh"]:
        run(f"act_{act}", X, y, n_classes, activation=act)
    for lr in [1e-2, 1e-4]:
        run(f"lr_{lr}", X, y, n_classes, lr=lr)
    for bs in [16, 256]:
        run(f"batch_{bs}", X, y, n_classes, batch_size=bs)
    # Regularisation, compared against the baseline
    run("reg_dropout0.2", X, y, n_classes, dropout=0.2)
    run("reg_weight_decay1e-3", X, y, n_classes, weight_decay=1e-3)
    run("reg_early_stop", X, y, n_classes, epochs=300, patience=20)


# Results of explore() (5-fold CV balanced accuracy, baseline = 1x64 ReLU,
# lr 1e-3, batch 64, 50 epochs, no regularisation):
#   baseline 94.11 | width 16/128/256: 93.94/94.25/94.15
#   layers 64x64 / 128x64 / 128x128x64: 94.20 / 94.30 / 93.94
#   leaky_relu 94.03 | tanh 93.95 | lr 1e-2 94.04, 1e-4 93.68 | batch 16 93.96, 256 94.12
#   dropout 0.2 94.10 | weight decay 1e-3 94.02 | early stopping 94.23
# Almost every change is within one std (~0.3%) of the baseline: the data is
# easy to separate, so a small network is already enough. lr 1e-4 is too slow
# to converge in 50 epochs; a 3rd hidden layer starts to overfit. Early
# stopping was the most useful regulariser (it also picks the epoch count).
FINAL = {**BASELINE, "hidden": (128, 64), "dropout": 0.2, "epochs": 300, "patience": 20}

EXPLORE = False  # set True to rerun the hyperparameter search (~45 min)
MODEL_PATH = "network.pt"


if __name__ == "__main__":
    X, y, classes, test = load_data()
    n_classes = len(classes)
    if EXPLORE:
        explore(X, y, n_classes)

    mean, sd = cross_validate(FINAL, X, y, n_classes, name="final")
    print(f"Final network (5-fold CV) BalAcc {mean*100:.2f} +-{sd*100:.2f}")

    # Train on all labelled data (10% held back only to decide when to stop)
    inner = make_folds(y, 10)
    val_idx, fit_idx = inner[0], np.concatenate(inner[1:])
    scaler = Standardiser().fit(X[fit_idx])
    model = train_model(scaler.transform(X[fit_idx]), y[fit_idx],
                        scaler.transform(X[val_idx]), y[val_idx], FINAL, n_classes,
                        log_dir="runs/final_full")

    # Save the weights with state_dict, then load them into a fresh model
    torch.save(model.state_dict(), MODEL_PATH)
    loaded = BeanMLP(X.shape[1], n_classes, FINAL["hidden"], FINAL["activation"],
                     FINAL["dropout"]).to(DEVICE)
    loaded.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    X_test = scaler.transform(test.to_numpy(dtype=np.float32))
    out = test.copy()
    out["Target"] = classes[predict(loaded, X_test)]
    out.to_csv("network.csv", index=False)
    print("Wrote network.csv:", out["Target"].value_counts().to_dict())
