import traceback

import numpy as np
from xgboost import XGBClassifier

import config
import data_processing
import modeling
import outputs
import visualization


def _tasks(dataset):
    all_rows = np.ones(len(dataset.y_multi), dtype=bool)
    yield "binary", dataset.y_binary, [1, 0], all_rows
    yield "binary_chi_mci", dataset.y_binary_chi_mci, [1, 0], dataset.chi_mci_mask
    yield "multiclass", dataset.y_multi, dataset.classes_multi, all_rows


def run_config(run_name, extra_exclude, dataset):
    X, feat_cols = data_processing.prepare_run(dataset.df_raw, run_name, extra_exclude)
    results = []
    for task, y_full, classes, mask in _tasks(dataset):
        out_dir = config.task_dir(run_name, task)
        out_dir.mkdir(parents=True, exist_ok=True)
        outputs.clean_stale_pdfs(out_dir)
        X_t, y_t = X[mask], y_full[mask]
        sex = dataset.sex_arr[mask] if dataset.sex_arr is not None else None
        rows = []
        print(f"\n  {run_name} / {config.TASK_DIRS[task]} (n={len(y_t)})")
        for mname, hyp, model, is_mlp in modeling.MODELS:
            try:
                r = visualization.make_report(
                    model, mname, hyp, X_t, y_t, feat_cols, task, classes,
                    is_mlp=is_mlp, is_xgb=isinstance(model, XGBClassifier),
                    sex_arr=sex, sex_male=dataset.sex_male, sex_female=dataset.sex_female,
                    out_dir=out_dir,
                )
                r["run"] = run_name
                rows.append(r)
            except Exception:
                print(f"  ERROR {mname} ({task}):")
                traceback.print_exc()
        results.extend(outputs.finalize_task(rows))
    return results


def main():
    dataset = data_processing.load_dataset()
    all_results = []
    for run_name, extra in config.RUN_CONFIGS:
        all_results.extend(run_config(run_name, extra, dataset))
    outputs.write_comparison_excel(all_results)
    outputs.publish_final_summary(all_results)
    outputs.remove_staging_pngs()
    print(f"\nDone -> {config.RESULTS_DIR}/")


if __name__ == "__main__":
    main()
