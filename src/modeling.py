import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    confusion_matrix, average_precision_score, log_loss,
)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC
from xgboost import XGBClassifier

import config

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

MLP_HYP = dict(hidden_layer_sizes=(128, 64), alpha=1e-4,
               learning_rate_init=1e-3, random_state=config.RANDOM_STATE)

MODELS = [
    ("LogisticRegression", "max_iter=2000, C=1.0, cw=balanced",
     LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced",
                        solver="lbfgs", random_state=config.RANDOM_STATE), False),
    ("RandomForest", "n_est=500, cw=bal_subsample",
     RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample",
                            random_state=config.RANDOM_STATE, n_jobs=-1), False),
    ("MLP_h128_64", "hidden=(128,64), alpha=1e-4, lr=1e-3",
     MLPClassifier(**MLP_HYP, max_iter=500), True),
    ("SVM_RBF", "C=1.0, kernel=rbf, cw=balanced",
     SVC(C=1.0, kernel="rbf", probability=True,
         class_weight="balanced", random_state=config.RANDOM_STATE), False),
    ("XGBoost", "n_est=300, depth=4, lr=0.05",
     XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                   subsample=0.8, colsample_bytree=0.8,
                   eval_metric="logloss", random_state=config.RANDOM_STATE, verbosity=0), False),
]


def cross_validate(model, X, y, cv_classes):
    cls_to_col = {c: i for i, c in enumerate(cv_classes)}
    y_pred_oof = np.empty(len(y), dtype=y.dtype)
    y_proba_oof = np.zeros((len(y), len(cv_classes)))
    fold_accs, fold_losses = [], []
    kf = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)
    for train_idx, test_idx in kf.split(X, y):
        m = clone(model)
        m.fit(X[train_idx], y[train_idx])
        y_pred_oof[test_idx] = m.predict(X[test_idx])
        proba = m.predict_proba(X[test_idx])
        for j, c in enumerate(m.classes_):
            col = cls_to_col.get(c)
            if col is not None:
                y_proba_oof[test_idx, col] = proba[:, j]
        fold_accs.append(accuracy_score(y[test_idx], y_pred_oof[test_idx]))
        try:
            fold_losses.append(log_loss(y[test_idx], proba, labels=list(m.classes_)))
        except Exception:
            fold_losses.append(np.nan)
    return y_pred_oof, y_proba_oof, {"type": "folds", "fold_acc": np.array(fold_accs),
                                     "fold_loss": np.array(fold_losses)}


def cross_validate_mlp(X, y, cv_classes):
    cls_to_col = {c: i for i, c in enumerate(cv_classes)}
    y_pred_oof = np.empty(len(y), dtype=y.dtype)
    y_proba_oof = np.zeros((len(y), len(cv_classes)))
    shape = (config.N_SPLITS, config.N_EPOCHS)
    tr_acc, val_acc = np.zeros(shape), np.zeros(shape)
    tr_loss, vl_loss = np.zeros(shape), np.zeros(shape)
    kf = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)
    for fi, (tv_idx, test_idx) in enumerate(kf.split(X, y)):
        X_tv, y_tv = X[tv_idx], y[tv_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=config.RANDOM_STATE)
        tr_sub, vl_sub = next(split.split(X_tv, y_tv))
        X_tr, y_tr = X_tv[tr_sub], y_tv[tr_sub]
        X_vl, y_vl = X_tv[vl_sub], y_tv[vl_sub]
        epoch_model = MLPClassifier(**MLP_HYP, max_iter=1, warm_start=True, tol=1e-12)
        for ep in range(config.N_EPOCHS):
            epoch_model.fit(X_tr, y_tr)
            tr_acc[fi, ep] = accuracy_score(y_tr, epoch_model.predict(X_tr))
            val_acc[fi, ep] = accuracy_score(y_vl, epoch_model.predict(X_vl))
            try:
                tr_loss[fi, ep] = log_loss(y_tr, epoch_model.predict_proba(X_tr))
                vl_loss[fi, ep] = log_loss(y_vl, epoch_model.predict_proba(X_vl))
            except Exception:
                pass
        y_pred_oof[test_idx] = epoch_model.predict(X_test)
        proba = epoch_model.predict_proba(X_test)
        for j, c in enumerate(epoch_model.classes_):
            col = cls_to_col.get(c)
            if col is not None:
                y_proba_oof[test_idx, col] = proba[:, j]
    curves = {
        "type": "epochs",
        "train_acc": tr_acc.mean(0), "train_acc_std": tr_acc.std(0),
        "val_acc": val_acc.mean(0), "val_acc_std": val_acc.std(0),
        "train_loss": tr_loss.mean(0), "train_loss_std": tr_loss.std(0),
        "val_loss": vl_loss.mean(0), "val_loss_std": vl_loss.std(0),
    }
    return y_pred_oof, y_proba_oof, curves


def fit_full(model, X, y):
    m = clone(model)
    m.fit(X, y)
    return m


def _native_importance(model, n_features):
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_.astype(float)
    if hasattr(model, "coef_"):
        return np.abs(model.coef_).mean(axis=0)
    return np.ones(n_features) / n_features


def compute_shap(model, X_data, task, n_background=50):
    n, p = X_data.shape
    native = _native_importance(model, p)
    fallback = (np.tile(native, (n, 1)), native)
    if not SHAP_AVAILABLE:
        return fallback
    background = shap.sample(X_data, min(n_background, n))
    chi_idx = 1 if task.startswith("binary") else 0

    def from_list(sv):
        mat = np.asarray(sv[chi_idx] if len(sv) > chi_idx else sv[0], float)
        mean_abs = np.mean([np.abs(np.asarray(s, float)).mean(axis=0) for s in sv], axis=0)
        return mat, mean_abs

    if isinstance(model, (RandomForestClassifier, XGBClassifier)):
        try:
            sv = shap.TreeExplainer(model).shap_values(X_data)
            if isinstance(sv, list):
                return from_list(sv)
            sv = np.asarray(sv, float)
            if sv.ndim == 3:
                return sv[:, :, chi_idx], np.abs(sv).mean(axis=(0, 2))
            return sv, np.abs(sv).mean(axis=0)
        except Exception:
            pass
    if isinstance(model, LogisticRegression):
        try:
            sv = shap.LinearExplainer(model, X_data).shap_values(X_data)
            if isinstance(sv, list):
                mat, mean_abs = from_list(sv)
            else:
                sv = np.asarray(sv, float)
                mat = sv[:, :, chi_idx] if sv.ndim == 3 else sv
                mean_abs = np.abs(sv).mean(axis=0) if sv.ndim == 2 else np.abs(sv).mean(axis=(0, 2))
            if mat.ndim == 2 and mat.shape[1] == p:
                return mat, mean_abs
        except Exception:
            pass
    try:
        n_eval = min(100, n)
        values = shap.PermutationExplainer(model.predict_proba, background)(X_data[:n_eval]).values
        if values.ndim == 3:
            mat = np.asarray(values[:, :, chi_idx], float)
            mean_abs = np.abs(values).mean(axis=(0, 2))
        else:
            mat = np.asarray(values, float)
            mean_abs = np.abs(mat).mean(axis=0)
        full = np.zeros((n, p))
        full[:n_eval] = mat
        full[n_eval:] = mat.mean(axis=0)
        return full, mean_abs
    except Exception:
        return fallback


def chi_column(cv_classes):
    return cv_classes.index(1) if 1 in cv_classes else 0


def _misclassified(y_true, y_pred, labels, names):
    return {names[i]: int(((y_pred != y_true) & (y_true == labels[i])).sum())
            for i in range(len(labels))}


def score_binary(y_true, y_pred, y_proba, cv_classes, class_names=("CHI", "Impaired"), neg_color=0):
    chi_c = chi_column(cv_classes)
    labels = [1, 0]
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    try:
        auroc = roc_auc_score(y_true, y_proba[:, chi_c])
    except Exception:
        auroc = 0.5
    try:
        auprc = average_precision_score(y_true, y_proba[:, chi_c])
    except Exception:
        auprc = 0.0
    return dict(
        acc=acc, auroc=auroc, auprc=auprc, cm=cm, classes=labels, class_names=list(class_names),
        class_colors=[config.C[1], config.C.get(neg_color, config.C[0])],
        prec=list(prec), rec=list(rec), f1=list(f1), spec=list(rec),
        p_chi=y_proba[:, chi_c],
        misclassified=_misclassified(y_true, y_pred, labels, class_names),
    )


def score_multiclass(y_true, y_pred, y_proba, classes, cv_classes):
    chi_c = chi_column(cv_classes)
    names = [config.CLASS_NAMES[c] for c in classes]
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=classes, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    spec = []
    for i in range(len(classes)):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - cm[i, :].sum() - fp
        spec.append(float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0)
    try:
        auroc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    except Exception:
        auroc = 0.0
    try:
        y_bin = label_binarize(y_true, classes=classes)
        auprc = float(np.mean([average_precision_score(y_bin[:, i], y_proba[:, i])
                               for i in range(len(classes))]))
    except Exception:
        auprc = 0.0
    return dict(
        acc=acc, auroc=auroc, auprc=auprc, cm=cm, classes=classes, class_names=names,
        prec=list(prec), rec=list(rec), f1=list(f1), spec=spec, p_chi=y_proba[:, chi_c],
        misclassified=_misclassified(y_true, y_pred, classes, names),
    )
