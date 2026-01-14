import os
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import mlflow
import mlflow.sklearn


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_path",
        type=str,
        default="preprocessing/creditscoring_preprocessing/creditscoring_preprocessed.csv",
    )
    p.add_argument("--target_col", type=str, default="target")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--n_estimators", type=int, default=300)
    p.add_argument("--max_depth", type=int, default=None)
    return p.parse_args()


def resolve_data_path(p: str) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    # coba relative ke current working dir
    if path.exists():
        return path
    # fallback: relative ke root repo (1 level di atas folder Membangun_model)
    repo_root = Path(__file__).resolve().parent.parent
    alt = repo_root / path
    return alt


def main():
    args = parse_args()

    # Optional: set tracking URI dari env (mis. DagsHub) jika tersedia
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    exp_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "creditscoring-MSML")
    mlflow.set_experiment(exp_name)

    data_path = resolve_data_path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {data_path}")

    df = pd.read_csv(data_path)

    if args.target_col not in df.columns:
        raise ValueError(
            f"Kolom target '{args.target_col}' tidak ditemukan. Kolom tersedia: {list(df.columns)}"
        )

    X = df.drop(columns=[args.target_col])
    y = df[args.target_col]

    # Pastikan preprocessing sudah selesai (fitur semua numerik)
    non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if non_numeric:
        raise ValueError(
            "Masih ada kolom non-numerik (preprocessing belum tuntas). "
            f"Kolom non-numerik: {non_numeric}"
        )

    stratify = y if y.nunique() <= 20 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=stratify,
    )

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
        n_jobs=-1,
    )

    # Basic: hanya autolog (tanpa mlflow.log_metric / mlflow.log_artifact)
    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="rf-autolog"):
        model.fit(X_train, y_train)
        test_score = model.score(X_test, y_test)
        print("Test score:", test_score)


if __name__ == "__main__":
    main()
