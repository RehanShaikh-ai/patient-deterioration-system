import os
import json
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, roc_auc_score,
    classification_report, confusion_matrix, fbeta_score
)
from src.logger import get_logger


def _sweep_thresholds(y_test, y_prob, thresholds, beta: float) -> tuple:
    """Return per-threshold metrics and the threshold maximising F-beta."""
    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        rows.append({
            "threshold": round(t, 2),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f_beta":    round(fbeta_score(y_test, y_pred, beta=beta, zero_division=0), 4),
        })

    best = max(rows, key=lambda r: r["f_beta"])
    return rows, best["threshold"]


def evaluate_models(trained: dict, cfg: dict, log_cfg: dict) -> tuple:
    logger = get_logger("evaluation", **log_cfg)
    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    t_min  = cfg.get("threshold_min", 0.1)
    t_max  = cfg.get("threshold_max", 0.9)
    t_step = cfg.get("threshold_step", 0.05)
    beta   = cfg.get("fbeta", 2)
    thresholds = np.arange(t_min, t_max + t_step, t_step)

    results = {}
    best_thresholds = {}

    for name, data in trained.items():
        model  = data["model"]
        X_test = data["X_test"]
        y_test = data["y_test"]
        y_prob = model.predict_proba(X_test)[:, 1]

        # --- threshold sweep ---
        sweep, best_t = _sweep_thresholds(y_test, y_prob, thresholds, beta)
        best_thresholds[name] = best_t

        sweep_path = os.path.join(output_dir, f"{name}_threshold_sweep.json")
        with open(sweep_path, "w") as f:
            json.dump(sweep, f, indent=2)

        logger.info(f"{name} — optimal threshold (F{beta}): {best_t}")
        logger.info(f"  Threshold sweep saved → {sweep_path}")

        # --- evaluate at optimal threshold ---
        y_pred = (y_prob >= best_t).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        metrics = {
            "threshold":  best_t,
            "precision":  round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall":     round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f_beta":     round(fbeta_score(y_test, y_pred, beta=beta, zero_division=0), 4),
            "roc_auc":    round(roc_auc_score(y_test, y_prob), 4),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        }
        results[name] = metrics

        logger.info(
            f"  {name}: precision={metrics['precision']} recall={metrics['recall']} "
            f"F{beta}={metrics['f_beta']} AUC={metrics['roc_auc']} "
            f"| TP={tp} FN={fn} FP={fp} TN={tn}"
        )

        report_path = os.path.join(output_dir, f"{name}_report.txt")
        with open(report_path, "w") as f:
            f.write(f"Optimal threshold: {best_t}\n\n")
            f.write(classification_report(y_test, y_pred))
            f.write(f"\nConfusion Matrix:\n  TN={tn}  FP={fp}\n  FN={fn}  TP={tp}\n")

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Evaluation summary → {summary_path}")

    # best model = highest recall, tie-break by F-beta
    best = max(results, key=lambda n: (results[n]["recall"], results[n]["f_beta"]))
    logger.info(
        f"Best model: {best} | recall={results[best]['recall']} "
        f"F{beta}={results[best]['f_beta']} threshold={results[best]['threshold']}"
    )
    return results, best, best_thresholds
