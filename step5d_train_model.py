import pandas as pd
import numpy as np
import os
import json

from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# CONFIG
# ============================================================

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

INPUT_FILE = BASE + r"\output\training_features.tsv"

MODEL_FILE = BASE + r"\code\business_entity_resolution\model\entity_matching_model.txt"

FEATURE_FILE = BASE + r"\code\business_entity_resolution\model\model_features.txt"

RANDOM_SEED = 42


# ============================================================
# LOAD FEATURES
# ============================================================

print("=" * 70)
print("STEP 5D - TRAIN ML MODEL")
print("=" * 70)

print("\nLoading training features...")

df = pd.read_csv(
    INPUT_FILE,
    sep="\t"
)

print("Rows:", len(df))
print("Columns:", len(df.columns))


# ============================================================
# REMOVE NON-FEATURE COLUMNS
# ============================================================

DROP_COLUMNS = [
    "source1_entity_id",
    "target_entity_id",
    "label"
]

X = df.drop(
    columns=DROP_COLUMNS
)

y = df["label"].astype(int)


# ============================================================
# CONVERT CATEGORICAL SOURCE
# ============================================================

# LightGBM can handle categorical columns,
# but using a simple numeric representation is
# easier for this first model.

if "target_source" in X.columns:

    X["target_source"] = (
        X["target_source"]
        .map({
            "S2": 0,
            "S3": 1
        })
        .fillna(-1)
    )


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.fillna(0)


# ============================================================
# FEATURE LIST
# ============================================================

feature_names = list(X.columns)

print()
print("Features used:")
for i, feature in enumerate(feature_names):
    print(i, feature)

print()
print("Total features:", len(feature_names))


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

print()
print("=" * 70)
print("CREATING TRAIN / VALIDATION SPLIT")
print("=" * 70)

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_SEED,
    stratify=y
)

print("Training rows:", len(X_train))
print("Validation rows:", len(X_valid))

print()
print("Training positives:", int(y_train.sum()))
print("Training negatives:", int((y_train == 0).sum()))

print()
print("Validation positives:", int(y_valid.sum()))
print("Validation negatives:", int((y_valid == 0).sum()))


# ============================================================
# CREATE MODEL
# ============================================================

print()
print("=" * 70)
print("TRAINING LIGHTGBM")
print("=" * 70)

model = LGBMClassifier(

    objective="binary",

    n_estimators=500,

    learning_rate=0.05,

    num_leaves=31,

    max_depth=-1,

    min_child_samples=30,

    subsample=0.8,

    colsample_bytree=0.8,

    reg_alpha=0.1,

    reg_lambda=0.5,

    random_state=RANDOM_SEED,

    n_jobs=-1,

    verbosity=-1
)


# ============================================================
# TRAIN
# ============================================================

model.fit(
    X_train,
    y_train
)

print("Model training complete.")


# ============================================================
# VALIDATION PROBABILITIES
# ============================================================

print()
print("=" * 70)
print("VALIDATION")
print("=" * 70)

probabilities = model.predict_proba(
    X_valid
)[:, 1]


# ============================================================
# TEST MULTIPLE THRESHOLDS
# ============================================================

print()
print("=" * 70)
print("THRESHOLD ANALYSIS")
print("=" * 70)

thresholds = [
    0.50,
    0.60,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95
]

best_threshold = 0.50
best_f05 = -1

for threshold in thresholds:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_valid,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_valid,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_valid,
        predictions,
        zero_division=0
    )

    # F0.5
    beta = 0.5

    if precision + recall > 0:

        f05 = (
            (1 + beta ** 2)
            * precision
            * recall
            /
            (
                beta ** 2 * precision
                + recall
            )
        )

    else:
        f05 = 0.0

    print(
        f"Threshold={threshold:.2f} "
        f"Precision={precision:.4f} "
        f"Recall={recall:.4f} "
        f"F1={f1:.4f} "
        f"F0.5={f05:.4f}"
    )

    if f05 > best_f05:

        best_f05 = f05
        best_threshold = threshold


# ============================================================
# BEST THRESHOLD RESULT
# ============================================================

best_predictions = (
    probabilities >= best_threshold
).astype(int)

best_precision = precision_score(
    y_valid,
    best_predictions,
    zero_division=0
)

best_recall = recall_score(
    y_valid,
    best_predictions,
    zero_division=0
)

best_f1 = f1_score(
    y_valid,
    best_predictions,
    zero_division=0
)


print()
print("=" * 70)
print("BEST VALIDATION RESULT")
print("=" * 70)

print(
    "Best threshold:",
    best_threshold
)

print(
    "Precision:",
    round(best_precision, 4)
)

print(
    "Recall:",
    round(best_recall, 4)
)

print(
    "F1:",
    round(best_f1, 4)
)

print(
    "F0.5:",
    round(best_f05, 4)
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print()
print("Confusion Matrix:")

cm = confusion_matrix(
    y_valid,
    best_predictions
)

print(cm)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()
print("Classification Report:")

print(
    classification_report(
        y_valid,
        best_predictions,
        zero_division=0
    )
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print()
print("=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

importance = pd.DataFrame({

    "feature": feature_names,

    "importance":
        model.feature_importances_

})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print(
    importance.to_string(index=False)
)


# ============================================================
# SAVE MODEL
# ============================================================

print()
print("=" * 70)
print("SAVING MODEL")
print("=" * 70)

model.booster_.save_model(
    MODEL_FILE
)

print(
    "Model saved:",
    MODEL_FILE
)


# ============================================================
# SAVE FEATURE NAMES
# ============================================================

with open(
    FEATURE_FILE,
    "w",
    encoding="utf-8"
) as f:

    for feature in feature_names:
        f.write(feature + "\n")


print(
    "Feature list saved:",
    FEATURE_FILE
)


# ============================================================
# SAVE THRESHOLD
# ============================================================

threshold_file = (
    BASE
    + r"\output\model_threshold.txt"
)

with open(
    threshold_file,
    "w"
) as f:

    f.write(
        str(best_threshold)
    )

print(
    "Threshold saved:",
    threshold_file
)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("STEP 5D COMPLETE")
print("=" * 70)

print()
print("Model:")
print(MODEL_FILE)

print()
print("Best threshold:")
print(best_threshold)

print()
print("Validation F0.5:")
print(round(best_f05, 4))

print()
print("=" * 70)
print("DONE")
print("=" * 70)

