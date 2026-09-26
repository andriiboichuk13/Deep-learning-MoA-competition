from sklearn.preprocessing import QuantileTransformer, OneHotEncoder
from sklearn.decomposition import PCA
import numpy as np




def make_features(trn_df, val_df, test_df, n_gene_pca=50, n_cell_pca=15, seed=42, return_transformers=False):
    # quantile transform: fit on training rows only, apply to all three
    g_cols = [c for c in trn_df.columns if c.startswith("g-")]
    c_cols = [c for c in trn_df.columns if c.startswith("c-")]
    qt_g = QuantileTransformer(output_distribution="normal", random_state=seed)
    qt_c = QuantileTransformer(output_distribution="normal", random_state=seed)

    g_tr = qt_g.fit_transform(trn_df[g_cols]);  g_va = qt_g.transform(val_df[g_cols]);  g_te = qt_g.transform(test_df[g_cols])
    c_tr = qt_c.fit_transform(trn_df[c_cols]);  c_va = qt_c.transform(val_df[c_cols]);  c_te = qt_c.transform(test_df[c_cols])

    # PCA: also fit on training rows only, on top of the quantile-transformed values
    pca_g = PCA(n_components=n_gene_pca, random_state=seed).fit(g_tr)
    pca_c = PCA(n_components=n_cell_pca, random_state=seed).fit(c_tr)

    pg_tr, pg_va, pg_te = pca_g.transform(g_tr), pca_g.transform(g_va), pca_g.transform(g_te)
    pc_tr, pc_va, pc_te = pca_c.transform(c_tr), pca_c.transform(c_va), pca_c.transform(c_te)

    # one-hot: fit on training rows only
    ohe = OneHotEncoder(handle_unknown="ignore")
    cat_tr = ohe.fit_transform(trn_df[["cp_time", "cp_dose"]]).toarray()
    cat_va = ohe.transform(val_df[["cp_time", "cp_dose"]]).toarray()
    cat_te = ohe.transform(test_df[["cp_time", "cp_dose"]]).toarray()

    # stick everything together: raw (transformed) features + PCA extras + one-hot
    X_tr = np.hstack([g_tr, c_tr, pg_tr, pc_tr, cat_tr]).astype(np.float32)
    X_va = np.hstack([g_va, c_va, pg_va, pc_va, cat_va]).astype(np.float32)
    X_te = np.hstack([g_te, c_te, pg_te, pc_te, cat_te]).astype(np.float32)
    if return_transformers:
        transformers = {"qt_g": qt_g, "qt_c": qt_c, "pca_g": pca_g, "pca_c": pca_c, "ohe": ohe}
        return X_tr, X_va, X_te, transformers
    return X_tr, X_va, X_te