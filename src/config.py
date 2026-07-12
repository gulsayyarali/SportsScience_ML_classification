from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

for _fname in (
    "SENDA_T1_Lars_v1_cleaned_features_and_samples.xlsx",
    "SENDA_T1_Lars_v2_cleaned_features_and_samples.xlsx",
    "SENDA_T1_Lars_v2_cleaned_feature_and_samples.xlsx",
):
    for _root in (DATA_DIR, PROJECT_ROOT.parent):
        if (_root / _fname).exists():
            DATA_FILE = _root / _fname
            break
    else:
        continue
    break
else:
    raise FileNotFoundError(f"No SENDA data file in {DATA_DIR} or {PROJECT_ROOT.parent}.")

RANDOM_STATE = 42
N_SPLITS = 5
N_EPOCHS = 30
SHAP_TOP_N = 20
BEST_MODEL_METRIC = "auroc"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.titlesize": 9,
    "figure.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "legend.framealpha": 0.75,
})

C = {1: "#43A047", 2: "#FDD835", 3: "#E53935", 0: "#FF7043"}
C_MALE, C_FEMALE = "#1565C0", "#AD1457"
C_TRAIN, C_VAL = "#1976D2", "#FF6F00"

CLASS_NAMES = {1: "CHI", 2: "pMCI", 3: "MCI"}
BINARY_NAMES = {1: "CHI", 0: "Impaired"}
BINARY_CHI_MCI_NAMES = {1: "CHI", 0: "MCI"}

BEESWARM_CMAP = LinearSegmentedColormap.from_list("beeswarm", ["#1E88E5", "#E53935"])

LABEL_DERIVED = {
    "MoCA_Gesamt_T1", "CERAD_Verbale_Fluessigkeit_T1",
    "CERAD_Phonematische_Fluessigkeit_T1", "CERAD_Phonematische_Fluessigkei_T1",
    "CERAD_LernenTotal_T1", "CERAD_SavingWort_T1", "CERAD_TrailBA_T1",
}
SKIP_COLS = {"Code", "MCI_CERAD_T1"}
NO_NORM_COLS = {"Code", "MCI_CERAD_T1", "sex"}
DEMOGRAPHIC_COLS = {"sex", "Jahre_AB", "Alter_Screening_T1"}

QUESTIONNAIRE_COLS = {
    "GDS_SCORE_T1", "Gesund_T1", "SozAkt_SCORE_T1", "Flei_Gesamt_T1",
    "BAECKE_INDEX_T1", "SOZ_SCORE_T1", "Tilburg_SCORE_T1", "NAI_SCORE_T1",
    "QL_SCORE_MEAN_T1", "FES_SCORE_T1",
}

DOMAIN_ORDER = [
    "Demographics", "Neuropsych", "Questionnaire",
    "Gait", "Fine motor", "Force & fitness", "EEG",
]
DOMAIN_ABBR = {
    "Demographics": "Demo", "Neuropsych": "Neuro", "Questionnaire": "Quest",
    "Gait": "Gait", "Fine motor": "FineMtr", "Force & fitness": "Force&Fit", "EEG": "EEG",
}
_DOMAIN_BLUES = LinearSegmentedColormap.from_list("domain_blues", ["#0D3B66", "#9EC9F2"])
DOMAIN_COLORS = {
    d: matplotlib.colors.to_hex(_DOMAIN_BLUES(i / (len(DOMAIN_ORDER) - 1)))
    for i, d in enumerate(DOMAIN_ORDER)
}

RUN_CONFIGS = [
    ("All_features", set()),
    ("wo_CERAD_demo", LABEL_DERIVED | DEMOGRAPHIC_COLS),
]

TASK_DIRS = {
    "binary": "Binary_CHI_vs_impaired",
    "binary_chi_mci": "Binary_CHI_vs_MCI",
    "multiclass": "multiclass_CHI_pMCI_MCI",
}

TASK_LABELS = {
    "binary": "Binary CHI vs Impaired",
    "binary_chi_mci": "Binary CHI vs MCI",
    "multiclass": "Multiclass CHI / pMCI / MCI",
}

MODEL_LABELS = {
    "LogisticRegression": "LR",
    "RandomForest": "RF",
    "MLP_h128_64": "MLP",
    "SVM_RBF": "SVM",
    "XGBoost": "XGBoost",
}

TASK_ORDER = ("binary", "binary_chi_mci", "multiclass")
MODEL_ORDER = ("LogisticRegression", "RandomForest", "MLP_h128_64", "SVM_RBF", "XGBoost")


def feature_domain(name):
    if name in DEMOGRAPHIC_COLS:
        return "Demographics"
    if name in LABEL_DERIVED or name == "DST_T1" or name.startswith("FL_"):
        return "Neuropsych"
    if name in QUESTIONNAIRE_COLS:
        return "Questionnaire"
    if name.startswith("norm_"):
        return "Gait"
    if name.startswith(("tap_", "plateau_", "peak_force", "velocity_",
                        "offphase", "time_to_peak", "pause_")):
        return "Fine motor"
    if name.startswith(("tot_", "var_", "steps_", "curls_")):
        return "Force & fitness"
    if name.startswith("log_"):
        return "EEG"
    return "Questionnaire"


def run_dir(run_name):
    return RESULTS_DIR / run_name


def task_dir(run_name, task):
    return run_dir(run_name) / TASK_DIRS[task]


def final_summary_dir(run_name, task):
    return RESULTS_DIR / "final_summary" / run_name / TASK_DIRS[task]
