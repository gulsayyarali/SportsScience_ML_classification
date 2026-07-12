"""Loading, cleaning, imputing, and normalising the SENDA T1 dataset."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

import config


def load_dataset():
    """Read the raw spreadsheet and derive the binary/multiclass targets."""
    print(f"Loading {config.DATA_FILE} ...")
    df_raw = pd.read_excel(config.DATA_FILE)
    df_raw = df_raw.dropna(subset=["MCI_CERAD_T1"])
    df_raw["MCI_CERAD_T1"] = df_raw["MCI_CERAD_T1"].astype(int)

    sex_arr = df_raw["sex"].values if "sex" in df_raw.columns else None
    sex_male, sex_female = 1, 2
    if sex_arr is not None:
        uniq = sorted(df_raw["sex"].dropna().unique())
        sex_male = uniq[0]
        sex_female = uniq[-1] if len(uniq) > 1 else None
        print(f"  sex unique: {uniq}  (interpreting {sex_male}=Male, {sex_female}=Female)")

    y_multi = df_raw["MCI_CERAD_T1"].values
    y_binary = (y_multi == 1).astype(int)
    classes_multi = sorted(np.unique(y_multi).tolist())

    # CHI vs MCI only — drop pMCI (label 2).
    chi_mci_mask = np.isin(y_multi, [1, 3])
    y_binary_chi_mci = (y_multi == 1).astype(int)

    print(f"  Raw shape: {df_raw.shape}")
    print(f"  Binary (CHI vs Impaired): "
          f"{dict(zip(*np.unique(y_binary, return_counts=True)))}")
    print(f"  Binary (CHI vs MCI, n={chi_mci_mask.sum()}): "
          f"{dict(zip(*np.unique(y_binary_chi_mci[chi_mci_mask], return_counts=True)))}")
    print(f"  Multi   : {dict(zip(*np.unique(y_multi, return_counts=True)))}")

    return SimpleNamespace(
        df_raw=df_raw, sex_arr=sex_arr, sex_male=sex_male, sex_female=sex_female,
        y_multi=y_multi, y_binary=y_binary, y_binary_chi_mci=y_binary_chi_mci,
        chi_mci_mask=chi_mci_mask, classes_multi=classes_multi,
    )


def _assemble_csv(feature_df, raw_df, save_cols):
    out = pd.DataFrame(index=raw_df.index)
    for col in save_cols:
        if col in config.SKIP_COLS:
            out[col] = raw_df[col].values
        elif col in feature_df.columns:
            out[col] = feature_df[col].values
        elif col in raw_df.columns:
            out[col] = raw_df[col].values
    return out


def prepare_run(df_raw, run_name, extra_exclude):
    feature_cols = [c for c in df_raw.columns if c not in config.SKIP_COLS and c not in extra_exclude]
    save_cols = list(df_raw.columns)

    print(f"\n{'=' * 60}\nRUN: {run_name}  |  features: {len(feature_cols)}\n{'=' * 60}")
    if extra_exclude:
        print(f"  excluded: {len(extra_exclude)} cols")

    imputer = SimpleImputer(strategy="median")
    x_imp = pd.DataFrame(imputer.fit_transform(df_raw[feature_cols]),
                         columns=feature_cols, index=df_raw.index)

    imp = config.RESULTS_DIR / f"SENDA_T1_{run_name}_imputed.csv"
    _assemble_csv(x_imp, df_raw, save_cols).to_csv(imp, index=False)
    print(f"  -> {imp.name}")

    x_norm = x_imp.copy()
    norm_cols = [c for c in feature_cols if c not in config.NO_NORM_COLS]
    if norm_cols:
        x_norm[norm_cols] = MinMaxScaler().fit_transform(x_imp[norm_cols])

    norm = config.RESULTS_DIR / f"SENDA_T1_{run_name}_normalized.csv"
    _assemble_csv(x_norm, df_raw, save_cols).to_csv(norm, index=False)
    print(f"  -> {norm.name}")

    return x_norm.values.astype(float), feature_cols
