import os
import json
from sklearn.metrics import precision_score, recall_score, roc_auc_score, classification_report
from src.logger import get_logger


def evaluate_models(trained: dict, cfg: dict, log_cfg: dict) -> tuple:
    logger = get_logger("evaluation", **log_cfg)
    output_dir = cfg["output_dir"]
    best_metric = cfg.get("best_model_metric", "recall")
    os.makedirs(output_dir, exist_ok=True)

    results = {}
    for name, data in trained.items():
        model = data["model"]
        X_test, y_test = data["X_test"], data["y_test"]
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        }
        results[name] = metrics
        logger.info(f"{name}: {metrics}")

        with open(os.path.join(output_dir, f"{name}_report.txt"), "w") as f:
            f.write(classification_report(y_test, y_pred))

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Evaluation summary saved to {output_dir}/summary.json")

    best = max(results, key=lambda n: results[n][best_metric])
    logger.info(f"Best model by {best_metric}: {best} ({results[best][best_metric]})")
    return results, best
