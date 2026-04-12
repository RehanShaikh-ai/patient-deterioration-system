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
    """Evaluate XGBoost primary + LR safety override at given thresholds."""
    # LR fires → positive; else XGBoost fires → positive; else negative
    y_pred = np.where(lr_prob >= lr_t, 1, np.where(xgb_prob >= xgb_t, 1, 0))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    precision  = round(precision_score(y_test, y_pred, zero_division=0), 4)
    recall     = round(recall_score(y_test, y_pred, zero_division=0), 4)
    f1         = round(f1_score(y_test, y_pred, zero_division=0), 4)
    alert_rate = round((tp + fp) / len(y_test), 4)
    logger.info(f"  Ensemble (XGB≥{xgb_t} | LR≥{lr_t}): "
                f"precision={precision} recall={recall} F1={f1} "
                f"alert_rate={alert_rate} | TP={tp} FP={fp} FN={fn} TN={tn}")
    return {
        "xgb_threshold": xgb_t, "lr_threshold": lr_t,
        "precision": precision, "recall": recall, "f1": f1,
        "alert_rate": alert_rate,
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
    probs_store = {}  # keep probs for ensemble eval

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

    # ── XGBoost focused threshold sweep (0.20–0.30) ───────────────────────────
    if "xgboost" in probs_store:
        y_test_xgb, xgb_prob = probs_store["xgboost"]
        ens_cfg = cfg.get("ensemble", {})
        xgb_t_min  = ens_cfg.get("xgb_threshold_min", 0.20)
        xgb_t_max  = ens_cfg.get("xgb_threshold_max", 0.30)
        xgb_t_step = ens_cfg.get("xgb_threshold_step", 0.02)
        lr_t       = ens_cfg.get("lr_threshold", 0.30)

        focused_thresholds = np.arange(xgb_t_min, xgb_t_max + xgb_t_step, xgb_t_step)
        focused_sweep, _ = _sweep_thresholds(y_test_xgb, xgb_prob, focused_thresholds, beta)

        logger.info(f"XGBoost focused threshold sweep ({xgb_t_min}–{xgb_t_max}):")
        for row in focused_sweep:
            logger.info(
                f"  t={row['threshold']:.2f} | precision={row['precision']} "
                f"recall={row['recall']} F1={row['f1']} alert_rate={row['alert_rate']}"
            )

        with open(os.path.join(output_dir, "xgboost_focused_sweep.json"), "w") as f:
            json.dump(focused_sweep, f, indent=2)

        # Select XGBoost threshold: best F-beta within alert_rate ≤ 0.40
        candidates = [r for r in focused_sweep if r["alert_rate"] <= 0.40]
        if not candidates:
            candidates = focused_sweep
        best_xgb_t = max(candidates, key=lambda r: (r["f_beta"], r["recall"]))["threshold"]
        best_thresholds["xgboost"] = best_xgb_t
        logger.info(f"Selected XGBoost threshold from focused sweep: {best_xgb_t}")

        # ── Ensemble evaluation ───────────────────────────────────────────────
        if "logistic_regression" in probs_store:
            _, lr_prob = probs_store["logistic_regression"]
            logger.info("Ensemble evaluation (XGBoost primary + LR safety override):")
            ensemble_results = []
            for xgb_t in focused_thresholds:
                xgb_t = round(float(xgb_t), 3)
                res = _eval_ensemble(y_test_xgb, lr_prob, xgb_prob, xgb_t, lr_t, beta, logger)
                ensemble_results.append(res)

            # Best ensemble: highest recall, alert_rate ≤ 0.40, tie-break F1
            best_ens = max(
                [r for r in ensemble_results if r["alert_rate"] <= 0.40] or ensemble_results,
                key=lambda r: (r["recall"], r["f1"])
            )
            results["ensemble"] = best_ens
            logger.info(
                f"Best ensemble config: XGB≥{best_ens['xgb_threshold']} | LR≥{best_ens['lr_threshold']} "
                f"→ recall={best_ens['recall']} precision={best_ens['precision']} "
                f"F1={best_ens['f1']} alert_rate={best_ens['alert_rate']}"
            )
            with open(os.path.join(output_dir, "ensemble_sweep.json"), "w") as f:
                json.dump(ensemble_results, f, indent=2)

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Evaluation summary → {output_dir}/summary.json")

    # Best individual model by recall then F-beta (exclude ensemble entry)
    individual = {k: v for k, v in results.items() if k != "ensemble"}
    best = max(individual, key=lambda n: (individual[n]["recall"], individual[n]["f_beta"]))
    logger.info(f"Best individual model: {best} | recall={results[best]['recall']} threshold={results[best]['threshold']}")
    return results, best, best_thresholds
