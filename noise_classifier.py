import argparse
from typing import Dict, List

import pandas as pd
 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


TARGET_COL = "Noise_level"
FEATURES = ["location", "DB_value", "Area_size", "Location_type"]


def ensure_required_columns(df: pd.DataFrame, required: List[str]):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    ensure_required_columns(df, [TARGET_COL] + FEATURES)

    # Ensure DB_value is numeric
    df["DB_value"] = pd.to_numeric(df["DB_value"], errors="coerce")
    return df


def build_preprocessor():
    numeric_features = ["DB_value"]
    categorical_features = ["location", "Area_size", "Location_type"]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )
    return preprocessor


def get_models(seed: int) -> Dict[str, object]:
    return {
        "log_reg": LogisticRegression(max_iter=1000, solver="lbfgs"),
        "decision_tree": DecisionTreeClassifier(random_state=seed, ccp_alpha=0.0),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=seed,
            min_samples_leaf=1,
            max_features="sqrt",
        ),
        "linear_svm": LinearSVC(random_state=seed),
    }

def main():
    parser = argparse.ArgumentParser(description="Noise level binary classification")
    parser.add_argument(
        "--data",
        required=True,
        help="Path to CSV file (expects columns: Noise_level, location, DB_value, Area_size, Location_type)",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Load data
    df = load_dataset(args.data)

    # Use only specified columns
    X = df[FEATURES].copy()
    y = df[TARGET_COL].astype(str).str.strip().str.lower()

    # Restrict to expected classes
    valid_classes = {"high", "moderate"}
    y_unique = set(y.unique())
    unknown = y_unique - valid_classes
    if unknown:
        raise ValueError(
            f"Unexpected target classes found: {unknown}. Expected exactly {valid_classes}."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    preprocessor = build_preprocessor()
    models = get_models(args.seed)

    all_metrics = {}

    for name, clf in models.items():
        pipe = Pipeline(steps=[("preprocess", preprocessor), ("clf", clf)], memory=None)
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="macro", zero_division=0
        )

        all_metrics[name] = {
            "accuracy": float(acc),
            "precision_macro": float(precision),
            "recall_macro": float(recall),
            "f1_macro": float(f1),
        }

        # Print classification report
        report = classification_report(y_test, y_pred)
        print(f"\n===== {name} =====")
        print(report)

        # Feature importances/coefficients

    # Print metrics summary
    best_model = max(all_metrics.items(), key=lambda kv: kv[1]["f1_macro"])[0]
    print("\n===== Summary (macro-averaged) =====")
    print(f"Test size: {args.test_size}")
    print(f"Seed: {args.seed}")
    print(f"Models evaluated: {', '.join(models.keys())}")
    for name, m in all_metrics.items():
        print(
            f"- {name}: acc={m['accuracy']:.3f}, precision={m['precision_macro']:.3f}, recall={m['recall_macro']:.3f}, f1={m['f1_macro']:.3f}"
        )
    print(f"Best model (by F1 macro): {best_model} ({all_metrics[best_model]['f1_macro']:.3f})")


if __name__ == "__main__":
    main()
