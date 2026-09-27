import numpy as np
import pandas as pd

eps = 1e-7

cnn_oof = pd.read_csv("cnn_oof.csv")
xgb_oof = pd.read_csv("xgb_oof.csv")
cdlm_oof = pd.read_csv("cdlm_oof.csv")
scored = pd.read_csv("Data/train_targets_scored_folds.csv")

target_cols = list(cnn_oof.columns[1:])
Y = scored[target_cols].values
p_cnn = cnn_oof[target_cols].values
p_xgb = xgb_oof[target_cols].values
p_cdlm = cdlm_oof[target_cols].values


def log_loss(y_true, y_pred):
    p = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))


print("cnn :", log_loss(Y, p_cnn))
print("xgb :", log_loss(Y, p_xgb))
print("cdlm:", log_loss(Y, p_cdlm))

best_loss = np.inf
best_w = None
step = 0.05

w1 = 0.0
while w1 <= 1.0:
    w2 = 0.0
    while w2 <= 1.0 - w1:
        w3 = 1.0 - w1 - w2
        blend = w1 * p_cnn + w2 * p_xgb + w3 * p_cdlm
        loss = log_loss(Y, blend)
        if loss < best_loss:
            best_loss = loss
            best_w = (w1, w2, w3)
        w2 += step
    w1 += step

print("\nbest weights (cnn, xgb, cdlm):", best_w)
print("best blend log loss:", best_loss)

cnn_test = pd.read_csv("cnn_test.csv")
xgb_test = pd.read_csv("xgb_test.csv")
cdlm_test = pd.read_csv("cdlm_test.csv")

w1, w2, w3 = best_w
blend_test = (w1 * cnn_test[target_cols].values
             + w2 * xgb_test[target_cols].values
             + w3 * cdlm_test[target_cols].values)
blend_test = np.clip(blend_test, eps, 1 - eps)

submission = pd.DataFrame(blend_test, columns=target_cols)
submission.insert(0, "sig_id", cnn_test["sig_id"].values)
submission.to_csv("ensemble_test.csv", index=False)
print("\nsaved ensemble_test.csv", submission.shape)