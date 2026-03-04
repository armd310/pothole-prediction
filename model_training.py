"""
Model Training & Evaluation for Montreal Pothole Repair Failure Prediction
==========================================================================
Takes model_ready.csv from feature_engineering.py and trains/evaluates
classifiers to predict repeat pothole repairs.

Key design decisions:
  - Cost-sensitive: false negatives (missed repeat) are more expensive
    than false positives (unnecessary extra attention)
  - Threshold tuning: optimizes decision threshold for best F1 on class 1
  - Compares multiple models with consistent evaluation
  - Outputs classification reports, feature importance, and confusion matrices

Output:
  - Console: full evaluation results
  - datasets/model_results/ folder with saved artifacts
"""

import logging
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Config ──────────────────────────────────────────────────────────────────

DATASETS_DIR = Path("datasets")
RESULTS_DIR = DATASETS_DIR / "model_results"
RESULTS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Cost ratio: how many times more costly is a false negative vs false positive
# Used for class_weight calculation
FN_COST_RATIO = 3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Load & Prepare ──────────────────────────────────────────────────────────

def load_and_prepare():
    """Load model_ready.csv, split features/target, identify column types."""
    log.info("=== Loading data ===")

    df = pd.read_csv(DATASETS_DIR / "model_ready.csv")
    log.info(f"Dataset shape: {df.shape}")
    log.info(f"Target distribution:\n{df['is_repeat'].value_counts(normalize=True)}")

    # Separate target
    y = df["is_repeat"]
    X = df.drop(columns=["is_repeat"])

    # Identify column types
    categorical_cols = []
    numeric_cols = []

    for col in X.columns:
        if X[col].dtype == "object" or col in ["equipment_type", "road_material"]:
            categorical_cols.append(col)
        else:
            numeric_cols.append(col)

    log.info(f"Numeric features ({len(numeric_cols)}): {numeric_cols}")
    log.info(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")

    # Train/test split — stratified to preserve class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    log.info(f"Train: {len(X_train):,}  Test: {len(X_test):,}")

    return X_train, X_test, y_train, y_test, numeric_cols, categorical_cols


# ── Preprocessing ───────────────────────────────────────────────────────────

def build_preprocessor(numeric_cols, categorical_cols):
    """Build a column transformer for numeric imputation+scaling + categorical encoding."""

    transformers = []

    if numeric_cols:
        # Impute then scale — median is robust to outliers in road age, traffic, etc.
        numeric_pipeline = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
        transformers.append(
            ("num", numeric_pipeline, numeric_cols)
        )

    if categorical_cols:
        categorical_pipeline = Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("encode", OneHotEncoder(handle_unknown="infrequent_if_exist", sparse_output=False)),
        ])
        transformers.append(
            ("cat", categorical_pipeline, categorical_cols)
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


# ── Models ──────────────────────────────────────────────────────────────────

def get_models():
    """Return dict of models to compare. All are cost-sensitive."""

    class_weight = {0: 1, 1: FN_COST_RATIO}

    models = {
        "Logistic Regression": LogisticRegression(
            class_weight=class_weight,
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_leaf=20,
            class_weight=class_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "HistGradient Boosting": HistGradientBoostingClassifier(
            max_iter=300,
            max_depth=8,
            learning_rate=0.1,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }

    return models


# ── Threshold Tuning ────────────────────────────────────────────────────────

def find_best_threshold(y_true, y_proba, optimize="f1"):
    """Find the decision threshold that maximizes F1 on the positive class."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    # F1 for each threshold
    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)

    best_idx = np.argmax(f1s)
    best_threshold = thresholds[min(best_idx, len(thresholds) - 1)]
    best_f1 = f1s[best_idx]

    return best_threshold, best_f1


# ── Evaluation ──────────────────────────────────────────────────────────────

def evaluate_model(name, model, preprocessor, X_train, X_test, y_train, y_test,
                   numeric_cols=None, categorical_cols=None):
    """Train model, find optimal threshold, evaluate on test set."""
    log.info(f"\n{'='*60}")
    log.info(f"Training: {name}")
    log.info(f"{'='*60}")

    # HistGradientBoosting handles NaN natively — skip imputation,
    # just encode categoricals with OrdinalEncoder (NaN-aware)
    if isinstance(model, HistGradientBoostingClassifier):
        from sklearn.preprocessing import OrdinalEncoder
        transformers = []
        if numeric_cols:
            transformers.append(("num", "passthrough", numeric_cols))
        if categorical_cols:
            transformers.append(
                ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                 categorical_cols)
            )
        hist_preprocess = ColumnTransformer(transformers=transformers, remainder="drop")
        pipe = Pipeline([
            ("preprocess", hist_preprocess),
            ("model", model),
        ])
    else:
        pipe = Pipeline([
            ("preprocess", preprocessor),
            ("model", model),
        ])

    # Fit
    pipe.fit(X_train, y_train)

    # Predict probabilities
    if hasattr(model, "predict_proba"):
        y_proba = pipe.predict_proba(X_test)[:, 1]
    else:
        y_proba = pipe.decision_function(X_test)

    # Default threshold (0.5)
    y_pred_default = (y_proba >= 0.5).astype(int)

    # Optimized threshold
    best_thresh, best_f1 = find_best_threshold(y_test, y_proba)
    y_pred_tuned = (y_proba >= best_thresh).astype(int)

    # ROC AUC
    try:
        auc = roc_auc_score(y_test, y_proba)
    except Exception:
        auc = float("nan")

    # Reports
    log.info(f"\n--- Default threshold (0.5) ---")
    report_default = classification_report(y_test, y_pred_default, output_dict=True)
    log.info(f"\n{classification_report(y_test, y_pred_default)}")

    log.info(f"\n--- Optimized threshold ({best_thresh:.3f}) ---")
    report_tuned = classification_report(y_test, y_pred_tuned, output_dict=True)
    log.info(f"\n{classification_report(y_test, y_pred_tuned)}")

    log.info(f"ROC AUC: {auc:.4f}")

    return {
        "name": name,
        "pipeline": pipe,
        "y_proba": y_proba,
        "y_pred_default": y_pred_default,
        "y_pred_tuned": y_pred_tuned,
        "best_threshold": best_thresh,
        "auc": auc,
        "report_default": report_default,
        "report_tuned": report_tuned,
    }


# ── Feature Importance ──────────────────────────────────────────────────────

def get_feature_importance(result, preprocessor, X_train):
    """Extract feature importance from tree-based models."""
    pipe = result["pipeline"]
    model = pipe.named_steps["model"]

    if not hasattr(model, "feature_importances_"):
        return None

    # Get transformed feature names
    feature_names = preprocessor.get_feature_names_out()
    # Clean up prefixes
    feature_names = [
        name.replace("num__", "").replace("cat__", "")
        for name in feature_names
    ]

    importances = model.feature_importances_
    fi = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    return fi


# ── Plotting ────────────────────────────────────────────────────────────────

def plot_results(results, preprocessor, X_train, y_test):
    """Generate comparison plots and save to results directory."""

    # 1. Model comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    model_names = [r["name"] for r in results]
    aucs = [r["auc"] for r in results]
    f1_defaults = [r["report_default"]["1"]["f1-score"] for r in results]
    f1_tuned = [r["report_tuned"]["1"]["f1-score"] for r in results]

    x = np.arange(len(model_names))
    width = 0.25

    ax.bar(x - width, aucs, width, label="ROC AUC", color="#2196F3")
    ax.bar(x, f1_defaults, width, label="F1 (default thresh)", color="#FF9800")
    ax.bar(x + width, f1_tuned, width, label="F1 (tuned thresh)", color="#4CAF50")

    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "model_comparison.png", dpi=150)
    plt.close(fig)
    log.info("Saved model_comparison.png")

    # 2. Confusion matrices for best model (tuned threshold)
    best = max(results, key=lambda r: r["report_tuned"]["1"]["f1-score"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ConfusionMatrixDisplay.from_predictions(
        y_test, best["y_pred_default"], ax=axes[0],
        display_labels=["Holds", "Repeat"], cmap="Blues"
    )
    axes[0].set_title(f"{best['name']} — Default (0.5)")

    ConfusionMatrixDisplay.from_predictions(
        y_test, best["y_pred_tuned"], ax=axes[1],
        display_labels=["Holds", "Repeat"], cmap="Blues"
    )
    axes[1].set_title(f"{best['name']} — Tuned ({best['best_threshold']:.3f})")

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "confusion_matrices.png", dpi=150)
    plt.close(fig)
    log.info("Saved confusion_matrices.png")

    # 3. Feature importance for best tree-based model
    fi = get_feature_importance(best, preprocessor, X_train)
    if fi is not None:
        fig, ax = plt.subplots(figsize=(10, 8))
        top_n = fi.head(20)
        ax.barh(range(len(top_n)), top_n["importance"].values, color="#2196F3")
        ax.set_yticks(range(len(top_n)))
        ax.set_yticklabels(top_n["feature"].values)
        ax.invert_yaxis()
        ax.set_xlabel("Importance")
        ax.set_title(f"Top 20 Features — {best['name']}")
        plt.tight_layout()
        fig.savefig(RESULTS_DIR / "feature_importance.png", dpi=150)
        plt.close(fig)
        log.info("Saved feature_importance.png")

        fi.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    log.info("Starting model training pipeline")

    # Load and prepare
    X_train, X_test, y_train, y_test, numeric_cols, categorical_cols = load_and_prepare()

    # Build preprocessor
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    # Fit preprocessor on training data (needed for feature names later)
    preprocessor.fit(X_train)

    # Train and evaluate all models
    models = get_models()
    results = []

    for name, model in models.items():
        result = evaluate_model(
            name, model, preprocessor,
            X_train, X_test, y_train, y_test,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
        )
        results.append(result)

    # Summary table
    log.info(f"\n{'='*60}")
    log.info("SUMMARY")
    log.info(f"{'='*60}")

    summary_rows = []
    for r in results:
        summary_rows.append({
            "Model": r["name"],
            "AUC": f"{r['auc']:.4f}",
            "F1 (default)": f"{r['report_default']['1']['f1-score']:.4f}",
            "F1 (tuned)": f"{r['report_tuned']['1']['f1-score']:.4f}",
            "Recall (tuned)": f"{r['report_tuned']['1']['recall']:.4f}",
            "Precision (tuned)": f"{r['report_tuned']['1']['precision']:.4f}",
            "Threshold": f"{r['best_threshold']:.3f}",
        })

    summary = pd.DataFrame(summary_rows)
    log.info(f"\n{summary.to_string(index=False)}")
    summary.to_csv(RESULTS_DIR / "model_summary.csv", index=False)

    # Plots
    plot_results(results, preprocessor, X_train, y_test)

    # Best model details
    best = max(results, key=lambda r: r["report_tuned"]["1"]["f1-score"])
    log.info(f"\nBest model: {best['name']} (F1={best['report_tuned']['1']['f1-score']:.4f} "
             f"at threshold={best['best_threshold']:.3f})")

    log.info(f"\nAll results saved to {RESULTS_DIR}")
    log.info("Done!")


if __name__ == "__main__":
    main()