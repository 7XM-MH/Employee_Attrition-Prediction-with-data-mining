"""Exécute tout le prétraitement : data/raw -> data/processed + models/.

Usage (depuis la racine du dépôt) :
    python scripts/run_preprocessing.py
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing import (  # noqa: E402
    TARGET, build_preprocessor, clean_data, iqr_outlier_report, load_raw, split_data,
)

RAW = ROOT / "data" / "raw" / "Employee_Attrition_DataSet.csv"
PROCESSED = ROOT / "data" / "processed"


def main(random_state: int = 42) -> dict:
    df_raw = load_raw(RAW)
    df, report = clean_data(df_raw)
    report["iqr_outliers"] = iqr_outlier_report(df).to_dict(orient="index")

    X_train, X_test, y_train, y_test = split_data(df, random_state=random_state)

    # Ajusté sur le train UNIQUEMENT (aucune fuite vers le test)
    pre = build_preprocessor()
    Xtr = pre.fit_transform(X_train)
    Xte = pre.transform(X_test)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    for old in PROCESSED.glob("*.csv"):
        old.unlink()
    Xtr.to_csv(PROCESSED / "X_train.csv", index=False)          # transformé (encodé + standardisé)
    Xte.to_csv(PROCESSED / "X_test.csv", index=False)
    X_train.to_csv(PROCESSED / "X_train_raw.csv", index=False)  # nettoyé, avant transformation
    X_test.to_csv(PROCESSED / "X_test_raw.csv", index=False)
    y_train.to_csv(PROCESSED / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED / "y_test.csv", index=False)

    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(pre, ROOT / "models" / "preprocessor.joblib")

    report.update(
        random_state=random_state,
        n_train=len(X_train), n_test=len(X_test),
        attrition_rate_train=round(float(y_train.mean()), 4),
        attrition_rate_test=round(float(y_test.mean()), 4),
        n_features_out=Xtr.shape[1],
        features_out=list(Xtr.columns),
    )
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "preprocessing_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    summary = pd.DataFrame({
        "metric": ["rows_raw", "rows_clean", "duplicates_removed", "n_train", "n_test",
                   "attrition_rate_train", "attrition_rate_test", "n_features_out"],
        "value": [report["rows_raw"], report["rows_clean"],
                  report["duplicate_rows_removed"] + report["duplicate_ids_removed"],
                  report["n_train"], report["n_test"], report["attrition_rate_train"],
                  report["attrition_rate_test"], report["n_features_out"]],
    })
    summary.to_csv(PROCESSED / "attrition_report.csv", index=False)
    return report


if __name__ == "__main__":
    r = main()
    print(f"Lignes brutes      : {r['rows_raw']}")
    print(f"Lignes nettoyées   : {r['rows_clean']}")
    print(f"Train / Test       : {r['n_train']} / {r['n_test']}")
    print(f"Taux de départ     : train={r['attrition_rate_train']:.2%}  test={r['attrition_rate_test']:.2%}")
    print(f"Variables en sortie: {r['n_features_out']}")
