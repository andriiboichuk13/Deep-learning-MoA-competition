import copy
from model import CNN
from CustomAdam import CustomAdam
import torch
import torch.nn as nn
import random
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from data_prep import make_features
from torch.optim.lr_scheduler import ExponentialLR
import wandb
import joblib

wandb.init(
    project="moa",
    name="CNN #1 train",
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

FINAL_EPOCHS = 11

criterion = nn.BCEWithLogitsLoss()

X_tr, _, X_te, transformers = make_features(train, train, test, return_transformers=True)
y_tr = scored[target_cols].values.astype(np.float32)

X_tr = torch.from_numpy(X_tr).to(device)
y_tr = torch.from_numpy(y_tr).to(device)

loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=256, shuffle=True, drop_last=True)

model = CNN(input_dim=X_tr.shape[1], output_dim=len(target_cols)).to(device)
optimizer = CustomAdam(model.parameters(), lr=2e-4, weight_decay=1e-2)
scheduler = ExponentialLR(optimizer, gamma = 0.95)

for epoch in range(FINAL_EPOCHS):
    model.train()
    train_loss = 0.
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        loss = criterion(model(x_batch), y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()
    scheduler.step()
    train_loss /= len(loader)
    grad_norms = {f"grad_norm/{name}": p.grad.norm().item()
                  for name, p in model.named_parameters() if p.grad is not None}

    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "lr": optimizer.param_groups[0]["lr"],
        **grad_norms,
    })
    print(f"epoch {epoch:3d}  train {train_loss:.5f}")

model.eval()
with torch.no_grad():
    X_te_t = torch.from_numpy(X_te).to(device)
    test_pred = torch.sigmoid(model(X_te_t)).cpu().numpy()

eps = 1e-7
test_pred = np.clip(test_pred, eps, 1 - eps)

pd.DataFrame(test_pred, columns=target_cols).assign(sig_id=test.sig_id.values)[["sig_id"] + target_cols] \
    .to_csv("cnn_test_full.csv", index=False)

torch.save(model.state_dict(), "cnn_full_model.pt")
joblib.dump(transformers, "cnn_transformers.joblib")
artifact = wandb.Artifact("cnn-full-model", type="model")
artifact.add_file("cnn_transformers.joblib")
artifact.add_file("cnn_full_model.pt")
wandb.log_artifact(artifact)

wandb.finish()