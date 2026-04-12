import os
import json
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, roc_auc_score,
    f1_score, classification_report, confusion_matrix, fbeta_score
)
from src.logger import get_logger


def _sweep_thresholds(y_test, y_prob, thresholds, beta: float) -> tuple:
    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        rows.append({
            "threshold": round(float(t), 3),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
            "f_beta":    round(fbeta_score(y_test, y_pred, beta=beta, zero_division=0), 4),
            "alert_rate": round(float(y_pred.sum()) / len(y_pred), 4),
        })
    best = max(rows, key=lambda r: r["f_beta"])
    return rows, best["threshold"]


def _eval_at_threshold(y_test, y_prob, threshold, beta):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    return {
        "threshold":  threshold,
        "precision":  round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":     round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1":         round(f1_score(y_test, y_pred, zero_division=0), 4),
        "f_beta":     round(fbeta_score(y_test, y_pred, beta=beta, zero_division=0), 4),
        "roc_auc":    round(roc_auc_score(y_test, y_prob), 4),
        "alert_rate": round((tp + fp) / len(y_test), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def _eval_ensemble(y_test, lr_prob, xgb_prob, xgb_t, lr_t, beta, logger):
    """
    Final decision logic:
      1. LR prob >= lr_t  → HIGH RISK
      2. XGB prob >= xgb_t → HIGH RISK
      3. else              → LOW RISK
    """
    y_pred = np.where((lr_prob >= lr_t) | (xgb_prob >= xgb_t), 1, 0)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    precision  = round(precision_score(y_test, y_pred, zero_division=0), 4)
    recall     = round(recall_score(y_test, y_pred, zero_division=0), 4)
    f1         = round(f1_score(y_test, y_pred, zero_division=0), 4)
    alert_rate = round((tp + fp) / len(y_test), 4)
    xgb_auc    = round(roc_auc_score(y_test, xgb_prob), 4)

    logger.info("=" * 55)
    logger.info("FINAL ENSEMBLE EVALUATION")
    logger.info(f"  Decision rule: LR≥{lr_t} OR XGB≥{xgb_t} → HIGH RISK")
    logger.info(f"  Recall     : {recall}")
    logger.info(f"  Precision  : {precision}")
    logger.info(f"  F1-score   : {f1}")
    logger.info(f"  XGB AUC    : {xgb_auc}")
    logger.info(f"  Alert rate : {alert_rate}  ({int(tp+fp)}/{len(y_test)} patients flagged)")
    logger.info(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    logger.info("=" * 55)

    return {
        "xgb_threshold": xgb_t,
        "lr_threshold":  lr_t,
        "precision":     precision,
        "recall":        recall,
        "f1":            f1,
        "xgb_auc":       xgb_auc,
        "alert_rate":    alert_rate,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


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
    probs_store = {}

    # ── Per-model evaluation ──────────────────────────────────────────────────
    for name, data in trained.items():
        model  = data["model"]
        X_test = data["X_test"]
        y_test = data["y_test"]
        y_prob = model.predict_proba(X_test)[:, 1]
        probs_store[name] = (y_test, y_prob)

        sweep, best_t = _sweep_thresholds(y_test, y_prob, thresholds, beta)
        best_thresholds[name] = best_t

        with open(os.path.join(output_dir, f"{name}_threshold_sweep.json"), "w") as f:
            json.dump(sweep, f, indent=2)

        metrics = _eval_at_threshold(y_test, y_prob, best_t, beta)
        results[name] = metrics
        cm = metrics["confusion_matrix"]

        logger.info(f"{name} — optimal threshold (F{beta}): {best_t}")
        logger.info(
            f"  precision={metrics['precision']} recall={metrics['recall']} "
            f"F1={metrics['f1']} F{beta}={metrics['f_beta']} AUC={metrics['roc_auc']} "
            f"alert_rate={metrics['alert_rate']} | "
            f"TP={cm['tp']} FP={cm['fp']} FN={cm['fn']} TN={cm['tn']}"
        )

        with open(os.path.join(output_dir, f"{name}_report.txt"), "w") as f:
            f.write(f"Optimal threshold: {best_t}\n\n")
            f.write(classification_report(y_test, (y_prob >= best_t).astype(int)))
            f.write(f"\nConfusion Matrix:\n  TN={cm['tn']}  FP={cm['fp']}\n  FN={cm['fn']}  TP={cm['tp']}\n")

    # ── Final ensemble evaluation at fixed thresholds ─────────────────────────
    ens_cfg = cfg.get("ensemble", {})
    xgb_t = ens_cfg.get("xgb_threshold", 0.20)
    lr_t  = ens_cfg.get("lr_threshold", 0.30)

    if "xgboost" in probs_store and "logistic_regression" in probs_store:
        y_test_shared, xgb_prob = probs_store["xgboost"]
        _, lr_prob = probs_store["logistic_regression"]
        ensemble_metrics = _eval_ensemble(y_test_shared, lr_prob, xgb_prob, xgb_t, lr_t, beta, logger)
        results["ensemble"] = ensemble_metrics

        with open(os.path.join(output_dir, "ensemble_report.txt"), "w") as f:
            f.write(f"Final Ensemble: LR≥{lr_t} OR XGB≥{xgb_t} → HIGH RISK\n\n")
            cm = ensemble_metrics["confusion_matrix"]
            f.write(f"Recall     : {ensemble_metrics['recall']}\n")
            f.write(f"Precision  : {ensemble_metrics['precision']}\n")
            f.write(f"F1-score   : {ensemble_metrics['f1']}\n")
            f.write(f"XGB AUC    : {ensemble_metrics['xgb_auc']}\n")
            f.write(f"Alert rate : {ensemble_metrics['alert_rate']}\n")
            f.write(f"TP={cm['tp']}  FP={cm['fp']}  FN={cm['fn']}  TN={cm['tn']}\n")

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Evaluation summary → {output_dir}/summary.json")

    individual = {k: v for k, v in results.items() if k != "ensemble"}
    best = max(individual, key=lambda n: (individual[n]["recall"], individual[n]["f_beta"]))
    logger.info(f"Best individual model: {best} | recall={results[best]['recall']} threshold={results[best]['threshold']}")
    return results, best, best_thresholds
