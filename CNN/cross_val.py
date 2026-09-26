import copy
from model import CNN
import torch
import torch.nn as nn
import random
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from data_prep import make_features
from CustomAdam import CustomAdam
from torch.optim.lr_scheduler import ExponentialLR
import wandb

wandb.init(
    project="moa",
    name="CNN #1 crossval",
    config={
        "lr": 2e-4,
        "weight_decay": 1e-2,
        "scheduler": "ExponentialLR",
        "gamma": 0.95,
        "channels": 256,
        "length": 16,
        "n_gene_pca": 50,
        "n_cell_pca": 15,
        "batch_size": 256,
        "patience": 15,
    }
)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train = pd.read_csv("Data/train_features_folds.csv")
scored = pd.read_csv("Data/train_targets_scored_folds.csv")
test = pd.read_csv("Data/test_features_trt.csv")
target_cols = list(scored.columns[2:])
Y = scored[target_cols].values
oof = np.zeros_like(Y, dtype=np.float32)
test_pred = np.zeros((len(test), len(target_cols)), dtype=np.float32)

criterion = nn.BCEWithLogitsLoss()
max_epochs, patience, min_delta = 100, 15, 1e-5

for fold in range(5):
    tr = train[train.fold != fold]
    val = train[train.fold == fold]
    X_tr, X_va, X_te = make_features(tr, val, test)
    y_tr = scored.loc[tr.index, target_cols].values.astype(np.float32)
    y_va = scored.loc[val.index, target_cols].values.astype(np.float32)

    X_tr = torch.from_numpy(X_tr).to(device)
    y_tr = torch.from_numpy(y_tr).to(device)
    X_val = torch.from_numpy(X_va).to(device)
    y_val = torch.from_numpy(y_va).to(device)

    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=256, shuffle=True, drop_last=True)

    best_loss, best_state, best_epoch, bad_epochs = float("inf"), None, 0, 0
    model = CNN(input_dim=X_tr.shape[1], output_dim=206).to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    optimizer = CustomAdam(model.parameters(), lr = 2e-4, weight_decay=1e-2)
    scheduler = ExponentialLR(optimizer, gamma=0.95)
    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()
        train_loss /= len(loader)
        grad_norms = {f"grad_norm/{name}": p.grad.norm().item() for name, p in model.named_parameters() if
                      p.grad is not None}
        wandb.log({"epoch": epoch, **grad_norms})

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val).item()

        print(f"epoch {epoch:3d}  train {train_loss:.5f}  val {val_loss:.5f}")

        wandb.log({
            "epoch": epoch,
            f"fold_{fold}/train_loss": train_loss,
            f"fold_{fold}/val_loss": val_loss,
            f"fold_{fold}/lr": optimizer.param_groups[0]["lr"],
        })

        if val_loss < best_loss - min_delta:
            best_loss, best_epoch, bad_epochs = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("Early stopping")
                break

    model.load_state_dict(best_state)
    print(f"fold {fold}: best epoch {best_epoch}, val loss {best_loss:.5f}")


    with torch.no_grad():
        model.eval()
        oof[val.index] = torch.sigmoid(model(X_val)).cpu().numpy()
        X_te_t = torch.from_numpy(X_te).to(device)
        test_pred += torch.sigmoid(model(X_te_t)).cpu().numpy() / 5

eps = 1e-7
oof_clipped = np.clip(oof, eps, 1 - eps)
cv_loss = -np.mean(Y * np.log(oof_clipped) + (1 - Y) * np.log(1 - oof_clipped))
print(f"\nCV log loss: {cv_loss:.5f}")

pd.DataFrame(oof, columns=target_cols).assign(sig_id=train.sig_id.values)[["sig_id"] + target_cols].to_csv("cnn_oof.csv", index=False)
pd.DataFrame(test_pred, columns=target_cols).assign(sig_id=test.sig_id.values)[["sig_id"] + target_cols].to_csv("cnn_test.csv", index=False)
wandb.log({"cv_log_loss": cv_loss})
wandb.finish()
