import shutil
from pathlib import Path

import pandas as pd

import config


def _best_row(df):
    cols = [config.BEST_MODEL_METRIC, "f1_macro"]
    return df.sort_values(cols, ascending=False).iloc[0]


def clean_stale_pdfs(out_dir):
    """Remove PDFs from models no longer in the pipeline (e.g. old HistGB runs)."""
    if not out_dir.is_dir():
        return
    for pdf in out_dir.glob("*.pdf"):
        if not any(model in pdf.name for model in config.MODEL_ORDER):
            pdf.unlink()


def dedupe_pdfs(rows):
    if not rows:
        return rows
    df = pd.DataFrame(rows)
    kept = []
    for _, group in df.groupby("model", sort=False):
        best = _best_row(group)
        for idx, row in group.iterrows():
            if idx != best.name:
                p = Path(row["pdf_path"])
                if p.exists():
                    p.unlink()
        kept.append(best.to_dict())
    return kept


def finalize_task(rows):
    return dedupe_pdfs(rows)


def publish_final_summary(all_rows):
    if not all_rows:
        return
    summary_root = config.RESULTS_DIR / "final_summary"
    if summary_root.exists():
        shutil.rmtree(summary_root)
    summary_root.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(all_rows)
    for task in config.TASK_ORDER:
        group = df[df["task"] == task]
        if group.empty:
            continue
        best = _best_row(group)
        dest = config.final_summary_dir(task)
        dest.mkdir(parents=True, exist_ok=True)
        model = best["model"]
        for key, suffix in (("p1_png", "_p1_overview.png"), ("p2_png", "_p2_evaluation.png")):
            src = Path(best.get(key, ""))
            if src.exists():
                shutil.copy2(src, dest / f"{config.MODEL_LABELS.get(model, model).lower()}{suffix}")
        pdf = Path(best["pdf_path"])
        if pdf.exists():
            shutil.copy2(pdf, dest / pdf.name)
        print(f"  final_summary/{config.TASK_DIRS[task]}: {model} (auroc={best['auroc']:.3f})")


def _comparison_row(row):
    out = {
        "task": config.TASK_LABELS.get(row["task"], row["task"]),
        "model": config.MODEL_LABELS.get(row["model"], row["model"]),
        "accuracy": round(row["acc"], 4),
        "roc_auc": round(row["auroc"], 4),
        "avg_precision": round(row["auprc"], 4),
        "f1_macro": round(row["f1_macro"], 4),
    }
    for i, name in enumerate(row["class_names"]):
        out[f"precision_{name}"] = round(row["prec"][i], 4)
        out[f"recall_{name}"] = round(row["rec"][i], 4)
        out[f"f1_{name}"] = round(row["f1"][i], 4)
        out[f"specificity_{name}"] = round(row["spec"][i], 4)
        out[f"misclassified_{name}"] = row["misclassified"][name]
    return out


def write_comparison_excel(all_rows):
    path = config.RESULTS_DIR / "model_comparison.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for run_name, _ in config.RUN_CONFIGS:
            rows = [r for r in all_rows if r["run"] == run_name]
            rows.sort(key=lambda r: (config.TASK_ORDER.index(r["task"]),
                                     config.MODEL_ORDER.index(r["model"])))
            pd.DataFrame([_comparison_row(r) for r in rows]).to_excel(
                writer, sheet_name=run_name, index=False)
    print(f"  -> {path.name}")


def remove_staging_pngs():
    for run_name, _ in config.RUN_CONFIGS:
        for task in config.TASK_DIRS:
            d = config.task_dir(run_name, task)
            if d.is_dir():
                for png in d.glob("*.png"):
                    png.unlink()
