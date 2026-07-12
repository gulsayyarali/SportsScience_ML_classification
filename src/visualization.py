"""All figure generation: per-model report PDFs (2 pages) and staging PNGs."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpecFromSubplotSpec
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.colorbar import ColorbarBase
import seaborn as sns
from scipy.interpolate import make_interp_spline
from sklearn.metrics import (
    accuracy_score, roc_curve, auc as sk_auc, precision_recall_curve,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize

import config
import modeling


def _spine_off(ax, sides=("top", "right")):
    for side in sides:
        ax.spines[side].set_visible(False)


def _safe_filename(s):
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(s))


def _title(acc, task, model_name, hyp, auroc, f1_macro):
    return f"{acc:.2f}_{task}_{model_name}  {hyp}  auc={auroc:.2f}  f1={f1_macro:.2f}"


def _base_filename(acc, task, model_name, auroc, f1_macro):
    return _safe_filename(f"{acc:.2f}_{task}_{model_name}_auc{auroc:.2f}_f1{f1_macro:.2f}")


def _png_filename(acc, task, model_name, auroc, f1_macro, page):
    return f"{_base_filename(acc, task, model_name, auroc, f1_macro)}_{page}.png"


def _pdf_filename(acc, task, model_name, auroc, f1_macro):
    return f"{_base_filename(acc, task, model_name, auroc, f1_macro)}.pdf"


def plot_sex_class_bar(ax, y, sex_arr, name_map, sex_male, sex_female):
    order = sorted(name_map.keys())[::-1]
    names = [name_map[k] for k in order]
    if sex_arr is not None:
        male_counts   = [(sex_arr[y == k] == sex_male).sum() for k in order]
        female_counts = [(sex_arr[y == k] == sex_female).sum() for k in order]
    else:
        male_counts, female_counts = [(y == k).sum() for k in order], [0] * len(order)

    ypos = np.arange(len(order))
    totals = [m + f for m, f in zip(male_counts, female_counts)]
    ax.barh(ypos, male_counts, color=config.C_MALE, alpha=0.85, height=0.45,
            edgecolor="none", label="Male")
    ax.barh(ypos, female_counts, left=male_counts, color=config.C_FEMALE, alpha=0.85,
            height=0.45, edgecolor="none", label="Female")
    for i, t in enumerate(totals):
        ax.text(t + max(totals) * 0.02, i, f"n={t}", va="center", fontsize=8)

    ax.set_yticks(ypos); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Count", fontsize=8)
    ax.set_title("Class Distribution by Sex", fontsize=9, fontweight="bold", pad=8)
    ax.set_xlim(0, max(totals) * 1.30)
    ax.legend(fontsize=7, loc="lower right")
    _spine_off(ax)


def plot_confusion_matrix(ax, cm, class_labels):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_labels, yticklabels=class_labels,
                ax=ax, cbar=False, linewidths=0.5, annot_kws={"size": 9})
    ax.set_xlabel("Predicted", fontsize=8)
    ax.set_ylabel("True", fontsize=8)
    ax.set_title("Confusion Matrix", fontsize=9, fontweight="bold", pad=8)
    ax.tick_params(labelsize=8)


def plot_classification_scatter(ax, y_true, p_chi, y_pred, task, metrics):
    """Per-sample P(CHI), sorted by class; misclassified points get a ring."""
    n = len(y_true)
    classes, class_names = metrics["classes"], metrics["class_names"]
    order = np.argsort(-y_true) if task.startswith("binary") else np.argsort(y_true)
    yt, ps, yp = y_true[order], p_chi[order], y_pred[order]

    for i in range(n):
        cl = int(yt[i])
        if task.startswith("binary") and "class_colors" in metrics:
            color = metrics["class_colors"][0 if cl == 1 else 1]
        else:
            color = config.C.get(cl, "grey")
        ax.scatter(i, ps[i], c=color, s=30, alpha=0.82, linewidths=0, zorder=2)
    for i in np.where(yp != yt)[0]:
        ax.scatter(i, ps[i], s=60, facecolors="none", edgecolors="black",
                   linewidths=1.1, zorder=3)

    ax.axhline(0.5, color="grey", linestyle="--", lw=0.9, alpha=0.8)
    ax.set_xlabel("Sample Index (OOF, sorted by class)", fontsize=8)
    ax.set_ylabel("P(CHI)", fontsize=8)
    ax.set_xlim(-3, n + 3)
    ax.tick_params(labelsize=7)

    acc = accuracy_score(yt, yp)
    if task == "multiclass":
        parts = [f"Misclassed {cn} {int(((yp != yt) & (yt == cl)).sum())}/{int((yt == cl).sum())}"
                 for cl, cn in zip(classes, class_names)]
        subtitle = f"Acc {acc:.2f} | " + " | ".join(parts)
    else:
        mc_chi, n_chi = int(((yp != yt) & (yt == 1)).sum()), int((yt == 1).sum())
        mc_neg, n_neg = int(((yp != yt) & (yt == 0)).sum()), int((yt == 0).sum())
        neg_name = class_names[1]
        subtitle = (f"Acc {acc:.2f} | Misclassed CHI {mc_chi}/{n_chi} | "
                    f"Misclassed {neg_name} {mc_neg}/{n_neg}")
    ax.set_title(f"Classification Result – {subtitle}", fontsize=8, fontweight="bold")

    if task.startswith("binary") and "class_colors" in metrics:
        patches = [mpatches.Patch(color=metrics["class_colors"][i], label=cn)
                   for i, cn in enumerate(class_names)]
    else:
        patches = [mpatches.Patch(color=config.C.get(cl, "grey"), label=cn)
                   for cl, cn in zip(classes, class_names)]
    patches.append(mpatches.Patch(facecolor="white", edgecolor="black", linewidth=1.5,
                                  label="Misclassified"))
    ax.legend(handles=patches, fontsize=7, loc="upper right", framealpha=0.7)


def beeswarm_row_offsets(shap_values, n_bins=100, row_half=0.44):
    """Stack same-bin points outward from the row centre (0, +1, -1, +2, ...)
    so the beeswarm forms a violin-like blob instead of a rectangular band."""
    n = len(shap_values)
    if n <= 1:
        return np.zeros(n)
    value_range = float(shap_values.max() - shap_values.min())
    if value_range < 1e-10:
        return np.zeros(n)

    bins = np.round(n_bins * (shap_values - shap_values.min()) / (value_range + 1e-8)).astype(int)
    rng = np.random.default_rng(42)
    tie_break = rng.random(n) * 1e-6
    order = np.argsort(bins.astype(float) + tie_break)

    offsets, layer, last_bin = np.zeros(n), 0, -1
    for i in order:
        if bins[i] != last_bin:
            layer, last_bin = 0, bins[i]
        offsets[i] = int(np.ceil(layer / 2)) * ((layer % 2) * 2 - 1)
        layer += 1

    max_offset = np.max(np.abs(offsets))
    if max_offset > 0:
        offsets = offsets * row_half / (max_offset + 1)
    return offsets


def _shrink_to_inset(ax_parent, height_frac=0.5):
    """Hide the parent axis and return a short, vertically-centred inset axis
    so the colourbar doesn't stretch across the whole SHAP row."""
    ax_parent.set_axis_off()
    y0 = (1.0 - height_frac) / 2.0
    return ax_parent.inset_axes([0.0, y0, 1.0, height_frac])


def _domain_colorbar(ax_cbar):
    cax = _shrink_to_inset(ax_cbar, height_frac=0.5)
    cmap = ListedColormap([config.DOMAIN_COLORS[d] for d in config.DOMAIN_ORDER])
    norm = BoundaryNorm(np.arange(len(config.DOMAIN_ORDER) + 1), cmap.N)
    cb = ColorbarBase(cax, cmap=cmap, norm=norm, orientation="vertical",
                      ticks=np.arange(len(config.DOMAIN_ORDER)) + 0.5)
    cb.set_ticklabels([config.DOMAIN_ABBR[d] for d in config.DOMAIN_ORDER])
    cb.ax.tick_params(labelsize=5.5, length=0)
    cb.outline.set_linewidth(0.5)
    cb.set_label("Domain", fontsize=7, labelpad=2)
    cax.yaxis.set_label_position("left")
    cax.yaxis.set_ticks_position("left")


def _feature_value_colorbar(ax_cbar):
    cax = _shrink_to_inset(ax_cbar, height_frac=0.3)
    cmap = ListedColormap([config.BEESWARM_CMAP(0.0), config.BEESWARM_CMAP(1.0)])
    norm = BoundaryNorm([0, 1, 2], cmap.N)
    cb = ColorbarBase(cax, cmap=cmap, norm=norm, orientation="vertical", ticks=[0.5, 1.5])
    cb.set_ticklabels(["Low", "High"])
    cb.ax.tick_params(labelsize=6.5, length=0)
    cb.outline.set_linewidth(0.5)
    cb.set_label("Feature value", fontsize=7, labelpad=2)


def plot_shap_pair(fig, gs_spec, shap_mat, shap_mean, X_vals, feat_names, top_n=config.SHAP_TOP_N):
    """SHAP bar + beeswarm, sharing one y-axis: domain colourbar, mirrored bar
    chart, feature names, beeswarm, and a low/high value colourbar — in that
    order left to right. Dashed grey row guides tie each name to its row."""
    n = min(top_n, len(feat_names))
    idx = np.argsort(shap_mean)[::-1][:n]
    names = [feat_names[i] for i in idx]
    vals = shap_mean[idx]
    ypos = np.arange(n)
    fontsize = max(4.0, min(5.2, 100.0 / max(n, 1)))
    bar_colors = [config.DOMAIN_COLORS[config.feature_domain(feat_names[i])] for i in idx]

    subgs = GridSpecFromSubplotSpec(1, 5, subplot_spec=gs_spec, wspace=0.06,
                                    width_ratios=[0.05, 0.95, 0.62, 1.35, 0.05])
    ax_domain_cb = fig.add_subplot(subgs[0])
    ax_bar       = fig.add_subplot(subgs[1])
    ax_labels    = fig.add_subplot(subgs[2], sharey=ax_bar)
    ax_bee       = fig.add_subplot(subgs[3], sharey=ax_bar)
    ax_value_cb  = fig.add_subplot(subgs[4])

    def row_guides(ax):
        for i in ypos:
            ax.axhline(i, color="#B0B0B0", lw=0.5, ls=(0, (3, 2)), zorder=0)

    ax_bar.set_ylim(-0.5, n - 0.5)
    ax_bar.invert_yaxis()

    ax_labels.set_xlim(0, 1)
    for spine in ax_labels.spines.values():
        spine.set_visible(False)
    for i, name in enumerate(names):
        ax_labels.text(0.5, i, name, fontsize=fontsize, ha="center", va="center",
                       clip_on=False, transform=ax_labels.transData)
    ax_labels.set_xticks([]); ax_labels.set_yticks([])
    ax_labels.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    row_guides(ax_bar)
    ax_bar.barh(ypos, vals, color=bar_colors, height=0.55, align="center",
               edgecolor="none", zorder=2)
    ax_bar.invert_xaxis()
    ax_bar.set_yticks([]); ax_bar.tick_params(axis="y", length=0, labelleft=False)
    _spine_off(ax_bar, ("top", "left"))
    xlabel = "Mean |SHAP|" if modeling.SHAP_AVAILABLE else "Feature Importance"
    ax_bar.set_xlabel(xlabel, fontsize=8)
    ax_bar.set_title(f"Top {n} – SHAP Importance", fontsize=9, fontweight="bold", pad=8)
    vmax = float(vals.max()) if vals.max() > 0 else 1.0
    for i, v in enumerate(vals):
        ax_bar.text(v + vmax * 0.02, i, f"{v:.4f}", va="center", ha="right",
                   fontsize=4.5, color="#333333", zorder=3)
    _domain_colorbar(ax_domain_cb)

    row_guides(ax_bee)
    for row_i, fi in enumerate(idx):
        sv = shap_mat[:, fi]
        fmin, fmax = X_vals[:, fi].min(), X_vals[:, fi].max()
        normed = (X_vals[:, fi] - fmin) / (fmax - fmin + 1e-10)
        is_high = (normed >= 0.5)[:, None]
        colors = np.where(is_high, np.array(config.BEESWARM_CMAP(1.0)),
                          np.array(config.BEESWARM_CMAP(0.0)))
        yj = row_i + beeswarm_row_offsets(sv)
        ax_bee.scatter(sv, yj, color=colors, s=14, alpha=1.0, linewidths=0,
                       zorder=2, rasterized=True)

    ax_bee.axvline(0, color="#333333", lw=1.0, zorder=1)
    ax_bee.set_yticks([]); ax_bee.tick_params(axis="y", length=0, labelleft=False)
    _spine_off(ax_bee, ("top", "right", "left"))
    lbl = "SHAP value (→ CHI)" if modeling.SHAP_AVAILABLE else "Contribution (→ CHI)"
    ax_bee.set_xlabel(lbl, fontsize=8)
    ax_bee.set_title(f"Top {n} – SHAP Beeswarm", fontsize=9, fontweight="bold", pad=8)
    _feature_value_colorbar(ax_value_cb)


def _smooth_line(ax, x, y, color, label, std=None, show_points=True):
    xs = np.linspace(x[0], x[-1], 300)
    k = min(3, len(x) - 1)
    spline = make_interp_spline(x, y, k=k)
    ax.plot(xs, spline(xs), color=color, lw=2.2, label=label)
    if show_points:
        ax.scatter(x, y, color=color, s=28, zorder=4, edgecolors="white", linewidths=0.6)
    if std is not None:
        smoothed_std = np.abs(make_interp_spline(x, std, k=k)(xs))
        ax.fill_between(xs, spline(xs) - smoothed_std, spline(xs) + smoothed_std,
                        alpha=0.18, color=color, linewidth=0)


def plot_curves(ax_acc, ax_loss, curves):
    if curves["type"] == "epochs":
        epochs = np.arange(1, config.N_EPOCHS + 1).astype(float)
        rows = [
            (ax_acc, "train_acc", "val_acc", "Accuracy",
             f"Accuracy over Epochs (mean±std, {config.N_SPLITS} folds)"),
            (ax_loss, "train_loss", "val_loss", "Log-Loss",
             f"Loss over Epochs (mean±std, {config.N_SPLITS} folds)"),
        ]
        for ax, train_key, val_key, ylabel, title in rows:
            _smooth_line(ax, epochs, curves[train_key], config.C_TRAIN, "Train",
                        std=curves[train_key + "_std"], show_points=False)
            _smooth_line(ax, epochs, curves[val_key], config.C_VAL, "Validation",
                        std=curves[val_key + "_std"], show_points=False)
            ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
            ax.set_title(title, pad=8); ax.legend(); _spine_off(ax)
    else:
        folds = np.arange(1, config.N_SPLITS + 1).astype(float)
        rows = [
            (ax_acc, "fold_acc", "Accuracy", f"Accuracy over {config.N_SPLITS} Folds", config.C_TRAIN),
            (ax_loss, "fold_loss", "Log-Loss", f"Log-Loss over {config.N_SPLITS} Folds", config.C_VAL),
        ]
        for ax, key, ylabel, title, color in rows:
            values = curves[key]
            mean_val = float(np.nanmean(values))
            _smooth_line(ax, folds, values, color, "Per-fold")
            ax.axhline(mean_val, color="#555", lw=1.3, linestyle="--",
                      label=f"Mean = {mean_val:.3f}", zorder=1)
            for xi, v in zip(folds, values):
                if not np.isnan(v):
                    ax.annotate(f"{v:.2f}", xy=(xi, v), xytext=(0, 7),
                               textcoords="offset points", ha="center", fontsize=6.5, color=color)
            ax.margins(y=0.20)
            ax.set_xticks(folds); ax.set_xticklabels([f"F{int(f)}" for f in folds])
            ax.set_xlabel("CV Fold"); ax.set_ylabel(ylabel)
            ax.set_title(title); ax.legend(); _spine_off(ax)


def plot_roc(ax, y_true, y_proba, task, metrics, cv_classes):
    if task.startswith("binary"):
        chi_c = modeling.chi_column(cv_classes)
        neg_c = 1 - chi_c
        chi_name, neg_name = metrics["class_names"]
        chi_col, neg_col = metrics.get("class_colors", [config.C[1], config.C[0]])
        pairs = [(y_true, y_proba[:, chi_c], chi_col, chi_name),
                 (1 - y_true, y_proba[:, neg_c], neg_col, neg_name)]
    else:
        y_bin = label_binarize(y_true, classes=metrics["classes"])
        pairs = [(y_bin[:, i], y_proba[:, i], config.C.get(cl, "grey"), cn)
                 for i, (cl, cn) in enumerate(zip(metrics["classes"], metrics["class_names"]))]

    for yt, score, color, label in pairs:
        try:
            fpr, tpr, _ = roc_curve(yt, score)
            area = sk_auc(fpr, tpr)
            ax.fill_between(fpr, tpr, alpha=0.20, color=color)
            ax.plot(fpr, tpr, lw=2, color=color, label=f"{label} AUC={area:.2f}")
        except Exception:
            pass

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.45)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("ROC-AUC"); ax.legend(); _spine_off(ax)


def plot_precision_recall(ax, y_true, y_proba, task, metrics, cv_classes):
    if task.startswith("binary"):
        chi_c = modeling.chi_column(cv_classes)
        neg_c = 1 - chi_c
        chi_name, neg_name = metrics["class_names"]
        chi_col, neg_col = metrics.get("class_colors", [config.C[1], config.C[0]])
        pairs = [(y_true, y_proba[:, chi_c], chi_col, chi_name),
                 (1 - y_true, y_proba[:, neg_c], neg_col, neg_name)]
    else:
        y_bin = label_binarize(y_true, classes=metrics["classes"])
        pairs = [(y_bin[:, i], y_proba[:, i], config.C.get(cl, "grey"), cn)
                 for i, (cl, cn) in enumerate(zip(metrics["classes"], metrics["class_names"]))]

    for yt, score, color, label in pairs:
        try:
            prec, rec, _ = precision_recall_curve(yt, score)
            ap = average_precision_score(yt, score)
            ax.fill_between(rec, prec, alpha=0.20, color=color)
            ax.plot(rec, prec, lw=2, color=color, label=f"{label} AP={ap:.2f}")
        except Exception:
            pass

    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall"); ax.legend(); _spine_off(ax)


def plot_per_class_bars(ax, values, class_labels, class_keys, title):
    x = np.arange(len(class_labels))
    colors = [config.C.get(k, "grey") for k in class_keys]
    bars = ax.bar(x, values, color=colors, edgecolor="none", width=0.42)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(class_labels, fontsize=8)
    ax.set_ylim(0, 1.18)
    ax.axhline(0.5, color="grey", linestyle="--", lw=0.8)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.tick_params(labelsize=7); _spine_off(ax)


def make_report(model, model_name, hyp_str, X, y, feat_names, task, classes,
                is_mlp, is_xgb, sex_arr, sex_male, sex_female, out_dir):
    """Train + cross-validate one model; save 2-page PDF + staging PNGs."""
    print(f"\n  [{task}] {model_name} ...")

    class_offset = min(classes) if (is_xgb and task == "multiclass") else 0
    y_fit = y - class_offset
    cv_classes = sorted(np.unique(y_fit).tolist())

    if is_mlp:
        y_pred_raw, y_proba, curves = modeling.cross_validate_mlp(X, y_fit, cv_classes)
    else:
        y_pred_raw, y_proba, curves = modeling.cross_validate(model, X, y_fit, cv_classes)
    y_pred = y_pred_raw + class_offset

    score_fn = modeling.score_binary if task.startswith("binary") else modeling.score_multiclass
    if task.startswith("binary"):
        if task == "binary_chi_mci":
            metrics = score_fn(y, y_pred, y_proba, cv_classes,
                               class_names=("CHI", "MCI"), neg_color=3)
        else:
            metrics = score_fn(y, y_pred, y_proba, cv_classes)
    else:
        metrics = score_fn(y, y_pred, y_proba, classes, cv_classes)
    f1_macro = float(np.mean(metrics["f1"]))
    print(f"    acc={metrics['acc']:.3f}  auroc={metrics['auroc']:.3f}  f1_macro={f1_macro:.3f}")

    print("    Computing SHAP/importance ...")
    full_model = modeling.fit_full(model, X, y_fit)
    shap_mat, shap_mean = modeling.compute_shap(full_model, X, task)

    suptitle = _title(metrics["acc"], task, model_name, hyp_str, metrics["auroc"], f1_macro)
    p1_name = _png_filename(metrics["acc"], task, model_name, metrics["auroc"], f1_macro, "p1")
    p2_name = _png_filename(metrics["acc"], task, model_name, metrics["auroc"], f1_macro, "p2")
    pdf_name = _pdf_filename(metrics["acc"], task, model_name, metrics["auroc"], f1_macro)
    class_keys = metrics["classes"]
    p1_path = out_dir / p1_name
    p2_path = out_dir / p2_name
    pdf_path = out_dir / pdf_name

    # PNG 1: class balance, confusion matrix, classification scatter, SHAP
    fig1 = plt.figure(figsize=(8.27, 11.69))
    fig1.suptitle(suptitle, fontsize=11, fontweight="bold", y=0.988)
    gs1 = gridspec.GridSpec(3, 2, figure=fig1, height_ratios=[1.35, 2.05, 2.0],
                            hspace=0.38, wspace=0.28, left=0.08, right=0.96, top=0.935, bottom=0.03)

    if task == "multiclass":
        name_map = config.CLASS_NAMES
    elif task == "binary_chi_mci":
        name_map = config.BINARY_CHI_MCI_NAMES
    else:
        name_map = config.BINARY_NAMES
    name_map = {k: v for k, v in name_map.items() if k in np.unique(y)}

    plot_sex_class_bar(fig1.add_subplot(gs1[0, 0]), y, sex_arr, name_map, sex_male, sex_female)
    plot_confusion_matrix(fig1.add_subplot(gs1[0, 1]), metrics["cm"], metrics["class_names"])
    plot_classification_scatter(fig1.add_subplot(gs1[1, :]), y, metrics["p_chi"], y_pred, task, metrics)
    plot_shap_pair(fig1, gs1[2, :], shap_mat, shap_mean, X, feat_names)

    fig1.savefig(p1_path, dpi=500, bbox_inches="tight", pad_inches=0.08)

    # PNG 2: training curves, ROC, PR, per-class specificity and F1
    fig2 = plt.figure(figsize=(8.27, 11.69))
    fig2.suptitle(suptitle + " _Evolution", fontsize=11, fontweight="bold", y=0.988)
    gs2 = gridspec.GridSpec(3, 2, figure=fig2, height_ratios=[1.4, 1.4, 1.0],
                            hspace=0.30, wspace=0.12, left=0.09, right=0.96, top=0.935, bottom=0.03)

    plot_curves(fig2.add_subplot(gs2[0, 0]), fig2.add_subplot(gs2[0, 1]), curves)
    plot_roc(fig2.add_subplot(gs2[1, 0]), y, y_proba, task, metrics, cv_classes)
    plot_precision_recall(fig2.add_subplot(gs2[1, 1]), y, y_proba, task, metrics, cv_classes)
    plot_per_class_bars(fig2.add_subplot(gs2[2, 0]), metrics["spec"], metrics["class_names"],
                        class_keys, "Specificity per Class")
    plot_per_class_bars(fig2.add_subplot(gs2[2, 1]), metrics["f1"], metrics["class_names"],
                        class_keys, "F1-Score per Class")

    fig2.savefig(p2_path, dpi=500, bbox_inches="tight")

    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig1, dpi=500, bbox_inches="tight", pad_inches=0.08)
        pdf.savefig(fig2, dpi=500, bbox_inches="tight")

    plt.close(fig1)
    plt.close(fig2)
    print(f"    Saved: {pdf_name}")

    return dict(
        task=task, model=model_name, run=None,
        acc=metrics["acc"], auroc=metrics["auroc"], auprc=metrics["auprc"], f1_macro=f1_macro,
        class_names=metrics["class_names"], prec=metrics["prec"], rec=metrics["rec"],
        f1=metrics["f1"], spec=metrics["spec"], misclassified=metrics["misclassified"],
        pdf_path=str(pdf_path), p1_png=str(p1_path), p2_png=str(p2_path),
    )
