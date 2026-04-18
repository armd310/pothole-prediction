"""TabNet Training & Evaluation Trains a TabNet classifier on the same model_ready.csv used by
model_training.py and appends its results to the existing model comparison.

Design decisions (kept consistent with model_training.py for fair comparison):
  - Same train/test split (same random_state, same test_size, stratified)
  - Same cost-sensitive weighting (3:1 FN/FP via class weights)
  - Same threshold tuning on precision-recall curve
  - Writes to the same datasets/model_results/ folder

TabNet-specific choices:
  - Categorical features passed as ordinal-encoded indices since it handles it internally
  - Numerical features: median-imputed + standard-scaled since it performs better on scaled data
  - Early stopping on validation AUC to prevent overfitting
  - CPU-compatible but GPU strongly recommended for reasonable runtime

Output:
  - Appends a row to model_summary.csv
  - Saves tabnet_training_curve.png and tabnet_feature_importance.csv
  - Updates model_comparison.png to include TabNet
"""

import logging
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    roc_auc_score,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

matplotlib.use("Agg")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# GLOBALS

DATASETS_DIR = Path("datasets")
RESULTS_DIR = DATASETS_DIR / "model_results"
RESULTS_DIR.mkdir(exist_ok=True)

# Match model_training.py
RANDOM_STATE = 42
TEST_SIZE = 0.2
FN_COST_RATIO = 3

# TabNet hyperparameters
TABNET_PARAMS = dict(
    n_d=32,  # decision layer width
    n_a=32,  # attention layer width
    n_steps=4,  # number of sequential attention steps
    gamma=1.3,  # feature reuse coefficient
    lambda_sparse=1e-4,  # sparsity regularization
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    scheduler_params=dict(step_size=10, gamma=0.9),
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    mask_type="entmax",  # sparser than softmax
    verbose=1,
    seed=RANDOM_STATE,
)

# Training config
MAX_EPOCHS = 100
BATCH_SIZE = 4096
VIRTUAL_BATCH_SIZE = 512  # for ghost batch normalization
PATIENCE = 10  # early stopping patience

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ==================================================
# LOADING AND PREPROCESSING
# ==============================================


def load_and_preprocess():
    """Load model_ready.csv and prepare TabNet-compatible arrays.

    Returns:
    X_train, X_val, X_test : np.ndarray
    y_train, y_val, y_test : np.ndarray
    cat_idxs : list[int] indices of categorical columns
    cat_dims : list[int] cardinality of each categorical col
    feature_names : list[str] column names in final order
    """
    log.info("=== Loading data ===")
    df = pd.read_csv(DATASETS_DIR / "model_ready.csv")
    log.info(f"Dataset shape: {df.shape}")

    y = df["is_repeat"].astype(int).values
    X = df.drop(columns=["is_repeat"])

    # Identify column types
    categorical_cols, numeric_cols = [], []
    for col in X.columns:
        if X[col].dtype == "object" or col in ["equipment_type", "road_material"]:
            categorical_cols.append(col)
        else:
            numeric_cols.append(col)

    log.info(f"Numeric ({len(numeric_cols)}): {numeric_cols}")
    log.info(f"Categorical ({len(categorical_cols)}): {categorical_cols}")

    # Numeric Cats: median-impute + scale
    num_imputer = SimpleImputer(strategy="median")
    num_scaler = StandardScaler()
    X_num = num_scaler.fit_transform(num_imputer.fit_transform(X[numeric_cols]))

    # Categorical Cats: ordinal encode and hill nan with
    # a sentinel string so the encoder treats it as a level
    X_cat_raw = X[categorical_cols].fillna("Unknown").astype(str)
    cat_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_cat = cat_encoder.fit_transform(X_cat_raw).astype(np.int64)
    # Shift any -1 values to 0 since TabNet needs non-negative indices
    # and add 1 to all cardinalities to make room
    X_cat = X_cat + 1
    cat_dims = [int(X_cat[:, i].max()) + 1 for i in range(X_cat.shape[1])]

    # stack all features
    X_all = np.hstack([X_num, X_cat]).astype(np.float32)
    cat_idxs = list(range(len(numeric_cols), len(numeric_cols) + len(categorical_cols)))
    feature_names = numeric_cols + categorical_cols

    # Same split as model_training.py
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_all,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    # Further carve a validation set for early stopping at 10% of train
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.10,
        random_state=RANDOM_STATE,
        stratify=y_train_full,
    )
    log.info(f"Train {len(X_train):,}  Val {len(X_val):,}  Test {len(X_test):,}")

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        cat_idxs,
        cat_dims,
        feature_names,
    )


# ============================================
# TRAIN
# ============================================


def train_tabnet(X_train, y_train, X_val, y_val, cat_idxs, cat_dims):
    """Fit a TabNetClassifier with cost-sensitive sample weights and early stopping on validation
    AUC.

    Arguments:
        Takes array X_train, y_train, X_val, y_val, cat_idxs, cat_dims as input.
    Returns:
        Returns trained TabNetClassifier.
    """
    # Cost-sensitive sample weights: minority class gets FN_COST_RATIO weight
    # TabNet-equivalent of class_weight={0: 1, 1: 3} in sklearn
    weights = np.where(y_train == 1, FN_COST_RATIO, 1).astype(np.float32)

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Training on device: {device_name}")

    model = TabNetClassifier(
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        cat_emb_dim=4,  # small embedding per categorical level
        device_name=device_name,
        **TABNET_PARAMS,
    )

    model.fit(
        X_train=X_train,
        y_train=y_train,
        eval_set=[(X_val, y_val)],
        eval_name=["val"],
        eval_metric=["auc"],
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        batch_size=BATCH_SIZE,
        virtual_batch_size=VIRTUAL_BATCH_SIZE,
        weights=weights,  # per-sample weighting
        drop_last=False,
    )
    return model


# ========================================
# THRESHOLD TUNING
# =========================================


def find_best_threshold(y_true, y_proba):
    """Identical to the threshold tuning in model_training.py.

    Find the decision threshold that maximizes F1 on the positive class.
    Arguments:
        Takes arrays y_true and y_proba as input.
    Returns:
        returns the best threshold and F1 score as floats.
    Example:
        best_thresh, best_f1 = find_best_threshold(y_test, y_proba)
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = int(np.argmax(f1s))
    best_thresh = thresholds[min(best_idx, len(thresholds) - 1)]
    return float(best_thresh), float(f1s[best_idx])


# ======================================
# EVALUATION
# ======================================


def evaluate_and_save(model, X_test, y_test, feature_names):
    """Score on test set, tune threshold, save artifacts.
    Arguments:
        Takes model and arrays X_test, y_test, feature_names as input.
    Returns:
        Summary Row and feature inportance DataFrame to add to model_summary.csv.
        Saves plots to RESULTS_DIR.
        - Training curve
        - Confusion matrix
        - Feature importance
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred_default = (y_proba >= 0.5).astype(int)

    best_t, _ = find_best_threshold(y_test, y_proba)
    y_pred_tuned = (y_proba >= best_t).astype(int)

    auc = roc_auc_score(y_test, y_proba)
    report_default = classification_report(y_test, y_pred_default, output_dict=True)
    report_tuned = classification_report(y_test, y_pred_tuned, output_dict=True)

    log.info("\n=== TabNet test results ===")
    log.info(f"ROC AUC: {auc:.4f}")
    log.info(
        f"\nDefault threshold (0.5):\n"
        f"{classification_report(y_test, y_pred_default)}"
    )
    log.info(
        f"\nTuned threshold ({best_t:.3f}):\n"
        f"{classification_report(y_test, y_pred_tuned)}"
    )

    # Append to model_summary.csv
    summary_row = {
        "Model": "TabNet",
        "AUC": f"{auc:.4f}",
        "F1 (default)": f"{report_default['1']['f1-score']:.4f}",
        "F1 (tuned)": f"{report_tuned['1']['f1-score']:.4f}",
        "Recall (tuned)": f"{report_tuned['1']['recall']:.4f}",
        "Precision (tuned)": f"{report_tuned['1']['precision']:.4f}",
        "Threshold": f"{best_t:.3f}",
    }
    summary_path = RESULTS_DIR / "model_summary.csv"
    if summary_path.exists():
        existing = pd.read_csv(summary_path)
        # Remove any prior TabNet row
        existing = existing[existing["Model"] != "TabNet"]
        updated = pd.concat([existing, pd.DataFrame([summary_row])], ignore_index=True)
    else:
        updated = pd.DataFrame([summary_row])
    updated.to_csv(summary_path, index=False)
    log.info(f"Updated {summary_path}")

    # Feature importance
    importances = model.feature_importances_
    fi = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)
    fi.to_csv(RESULTS_DIR / "tabnet_feature_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    top = fi.head(20)
    ax.barh(range(len(top)), top["importance"].values, color="#9C27B0")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"].values)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title("Top 20 Features — TabNet")
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "tabnet_feature_importance.png", dpi=150)
    plt.close(fig)

    # Training curve
    history = model.history
    if "val_auc" in history.history:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(history.history["val_auc"], label="val AUC", color="#9C27B0")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("AUC")
        ax.set_title("TabNet validation AUC over training")
        ax.legend()
        plt.tight_layout()
        fig.savefig(RESULTS_DIR / "tabnet_training_curve.png", dpi=150)
        plt.close(fig)

    # Confusion matrix
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred_default,
        ax=axes[0],
        display_labels=["Holds", "Repeat"],
        cmap="Purples",
    )
    axes[0].set_title("TabNet — Default (0.5)")
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred_tuned,
        ax=axes[1],
        display_labels=["Holds", "Repeat"],
        cmap="Purples",
    )
    axes[1].set_title(f"TabNet — Tuned ({best_t:.3f})")
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "tabnet_confusion_matrices.png", dpi=150)
    plt.close(fig)

    return summary_row, fi


# ==============================================
# NEW COMPARISON PLOT
# =============================================


def regenerate_comparison_plot():
    """Redraw model_comparison.png from the updated model_summary.csv so all four models appear on
    the same bar chart.

    Returns:
        Saves model_comparison.png to RESULTS_DIR.
    """
    summary = pd.read_csv(RESULTS_DIR / "model_summary.csv")

    fig, ax = plt.subplots(figsize=(11, 5))
    names = summary["Model"].tolist()
    aucs = summary["AUC"].astype(float).values
    f1_def = summary["F1 (default)"].astype(float).values
    f1_tuned = summary["F1 (tuned)"].astype(float).values

    x = np.arange(len(names))
    w = 0.25
    ax.bar(x - w, aucs, w, label="ROC AUC", color="#2196F3")
    ax.bar(x, f1_def, w, label="F1 (default thresh)", color="#FF9800")
    ax.bar(x + w, f1_tuned, w, label="F1 (tuned thresh)", color="#4CAF50")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title("Model Comparison (incl. TabNet)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "model_comparison.png", dpi=150)
    plt.close(fig)
    log.info("Regenerated model_comparison.png with TabNet")


# ============================================
# MAIN
# ===========================================


def main():
    log.info("Starting TabNet training pipeline")
    # Load and preprocess data
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        cat_idxs,
        cat_dims,
        feature_names,
    ) = load_and_preprocess()
    # Train tabnet model
    model = train_tabnet(X_train, y_train, X_val, y_val, cat_idxs, cat_dims)
    summary_row, fi = evaluate_and_save(model, X_test, y_test, feature_names)
    # Print feature importance
    log.info("\nTop 10 features (TabNet):")
    log.info(f"\n{fi.head(10).to_string(index=False)}")
    # Regenrate comparison plot
    regenerate_comparison_plot()
    # Summary log
    log.info(f"\nTabNet result: {summary_row}")
    log.info("Done.")


if __name__ == "__main__":
    main()
