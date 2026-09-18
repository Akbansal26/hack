"""
Baseline early-warning classifier: RandomForest on engineered rolling-window
features, split BY MACHINE (not by row) so the test set is fully unseen
equipment -- the realistic evaluation setup for predictive maintenance.

Also evaluates a trivial "naive threshold" baseline (vibration z-score alarm)
so the ML model's lift is visible, and computes lead time: how many hours
before the actual failure the model raises its first alert per machine.
"""
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

RNG_SEED = 7
TEST_FRACTION = 0.3
ALERT_THRESHOLD = 0.5


def machine_split(feat: pd.DataFrame, test_fraction=TEST_FRACTION, seed=RNG_SEED):
    machines = np.array(sorted(feat["machine_id"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(machines)
    n_test = max(1, int(len(machines) * test_fraction))
    test_machines = set(machines[:n_test])
    train_mask = ~feat["machine_id"].isin(test_machines)
    return feat[train_mask].copy(), feat[~train_mask].copy(), test_machines


def get_feature_columns(feat: pd.DataFrame):
    drop = {"machine_id", "timestamp", "step", "failure_event", "label"}
    return [c for c in feat.columns if c not in drop]


def naive_baseline_scores(feat: pd.DataFrame) -> np.ndarray:
    """Simple physics-style rule: alert if vibration_mean_24 is > 2.5 std above
    that machine's own first-72h healthy baseline. No ML at all."""
    feat = feat.reset_index(drop=True)
    scores = np.zeros(len(feat))
    for mid, sub in feat.groupby("machine_id"):
        healthy = sub[sub["step"] < 72]["vibration_mean_24"]
        if len(healthy) < 5:
            base_mean, base_std = sub["vibration_mean_24"].mean(), sub["vibration_mean_24"].std() + 1e-6
        else:
            base_mean, base_std = healthy.mean(), healthy.std() + 1e-6
        z = (sub["vibration_mean_24"] - base_mean) / base_std
        scores[sub.index.to_numpy()] = (z > 2.5).astype(float)
    return scores


def lead_time_hours(feat: pd.DataFrame, alert_col: str, summary: pd.DataFrame) -> dict:
    """For each failing test machine, hours between first alert and failure step."""
    results = {}
    fail_steps = summary.set_index("machine_id")["failure_step"].to_dict()
    for mid, sub in feat.groupby("machine_id"):
        if mid not in fail_steps or pd.isna(fail_steps[mid]):
            continue
        alerts = sub[sub[alert_col] == 1]
        if len(alerts) == 0:
            results[mid] = None  # missed entirely
        else:
            first_alert_step = alerts["step"].min()
            results[mid] = float(fail_steps[mid] - first_alert_step)
    return results


def evaluate(y_true, y_pred, y_proba=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    out = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "false_alarm_rate": false_alarm_rate,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }
    if y_proba is not None and len(set(y_true)) > 1:
        out["roc_auc"] = roc_auc_score(y_true, y_proba)
    return out


def main():
    feat = pd.read_csv("features.csv")
    summary = pd.read_csv("machine_summary.csv")
    train, test, test_machines = machine_split(feat)
    feature_cols = get_feature_columns(feat)

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=5,
        class_weight="balanced_subsample", random_state=RNG_SEED, n_jobs=-1,
    )
    clf.fit(train[feature_cols], train["label"])

    test = test.copy()
    test["rf_proba"] = clf.predict_proba(test[feature_cols])[:, 1]
    test["rf_alert"] = (test["rf_proba"] >= ALERT_THRESHOLD).astype(int)
    test["naive_alert"] = naive_baseline_scores(test)

    rf_metrics = evaluate(test["label"], test["rf_alert"], test["rf_proba"])
    naive_metrics = evaluate(test["label"], test["naive_alert"])

    rf_lead = lead_time_hours(test, "rf_alert", summary)
    naive_lead = lead_time_hours(test, "naive_alert", summary)

    def lead_stats(lead_dict):
        vals = [v for v in lead_dict.values() if v is not None]
        missed = sum(1 for v in lead_dict.values() if v is None)
        return {
            "mean_lead_hours": float(np.mean(vals)) if vals else None,
            "median_lead_hours": float(np.median(vals)) if vals else None,
            "n_failing_machines": len(lead_dict),
            "n_missed": missed,
        }

    # feature importance for the "signals that changed before the alert"
    importances = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)

    # per-machine risk ranking = latest known risk score for each test+train machine
    feat_all_scored = feat.copy()
    feat_all_scored["rf_proba"] = clf.predict_proba(feat_all_scored[feature_cols])[:, 1]
    latest = feat_all_scored.sort_values("step").groupby("machine_id").tail(1)
    ranking = latest[["machine_id", "step", "rf_proba"]].merge(
        summary[["machine_id", "will_fail"]], on="machine_id", how="left"
    ).sort_values("rf_proba", ascending=False)

    # compute threshold evaluation curves for dynamic dashboard exploration
    threshold_curves = []
    for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        th_alert = (test["rf_proba"] >= thresh).astype(int)
        th_eval = evaluate(test["label"], th_alert, test["rf_proba"])
        test_tmp = test.copy()
        test_tmp["th_alert"] = th_alert
        th_lead = lead_stats(lead_time_hours(test_tmp, "th_alert", summary))
        threshold_curves.append({
            "threshold": thresh,
            "precision": th_eval["precision"],
            "recall": th_eval["recall"],
            "f1": th_eval["f1"],
            "false_alarm_rate": th_eval["false_alarm_rate"],
            "mean_lead_hours": th_lead["mean_lead_hours"],
            "median_lead_hours": th_lead["median_lead_hours"],
            "n_missed": th_lead["n_missed"],
        })

    import os
    os.makedirs("results", exist_ok=True)

    results = {
        "n_train_machines": int(train["machine_id"].nunique()),
        "n_test_machines": int(test["machine_id"].nunique()),
        "test_machines": sorted(test_machines),
        "random_forest": rf_metrics,
        "naive_threshold": naive_metrics,
        "rf_lead_time": lead_stats(rf_lead),
        "naive_lead_time": lead_stats(naive_lead),
        "threshold_curves": threshold_curves,
        "top_features": importances.head(15).to_dict(),
    }
    with open("baseline_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    with open(os.path.join("results", "baseline_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    test.to_csv("test_scored.csv", index=False, encoding="utf-8")
    test.to_csv(os.path.join("results", "test_scored.csv"), index=False, encoding="utf-8")
    
    ranking.to_csv("machine_risk_ranking.csv", index=False, encoding="utf-8")
    ranking.to_csv(os.path.join("results", "machine_risk_ranking.csv"), index=False, encoding="utf-8")
    
    feat_all_scored[["machine_id", "step", "timestamp", "rf_proba", "label"]].to_csv(
        "all_scored.csv", index=False, encoding="utf-8"
    )

    print("Baseline Random Forest evaluated successfully:")
    print("RF Metrics:", json.dumps(results["random_forest"], indent=2))
    print("Naive Metrics:", json.dumps(results["naive_threshold"], indent=2))
    print("RF Lead Time:", json.dumps(results["rf_lead_time"], indent=2))


if __name__ == "__main__":
    main()
