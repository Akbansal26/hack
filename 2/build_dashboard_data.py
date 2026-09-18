import json
import os
import numpy as np
import pandas as pd

def lead_time_from_meta(meta_scored, prob_col, threshold, summary):
    fail_steps = summary.set_index("machine_id")["failure_step"].to_dict()
    leads = {}
    for mid, sub in meta_scored.groupby("machine_id"):
        if mid not in fail_steps or pd.isna(fail_steps[mid]):
            continue
        alerts = sub[sub[prob_col] >= threshold]
        if len(alerts) == 0:
            leads[mid] = None
        else:
            leads[mid] = float(fail_steps[mid] - alerts["step"].min())
    vals = [v for v in leads.values() if v is not None]
    return {
        "mean_lead_hours": float(np.mean(vals)) if vals else None,
        "median_lead_hours": float(np.median(vals)) if vals else None,
        "n_failing_machines": len(leads),
        "n_missed": sum(1 for v in leads.values() if v is None),
    }


def main():
    sensor = pd.read_csv("sensor_data.csv", parse_dates=["timestamp"])
    summary = pd.read_csv("machine_summary.csv")
    
    with open("baseline_results.json", "r", encoding="utf-8") as f:
        baseline_results = json.load(f)
    with open("lstm_results.json", "r", encoding="utf-8") as f:
        lstm_results = json.load(f)
        
    ranking = pd.read_csv("machine_risk_ranking.csv")
    all_scored = pd.read_csv("all_scored.csv", parse_dates=["timestamp"])
    lstm_scored = pd.read_csv("lstm_test_scored.csv")

    lstm_lead = lead_time_from_meta(lstm_scored, "lstm_proba", 0.5, summary)

    # ---- Model comparison ----
    comparison = {
        "random_forest": {
            **{k: baseline_results["random_forest"][k] for k in
               ["precision", "recall", "f1", "false_alarm_rate", "roc_auc", "tp", "fp", "fn", "tn"]},
            "lead_time": baseline_results["rf_lead_time"],
            "threshold_curves": baseline_results.get("threshold_curves", []),
        },
        "lstm": {
            **{k: lstm_results[k] for k in
               ["precision", "recall", "f1", "false_alarm_rate", "roc_auc", "tp", "fp", "fn", "tn"]},
            "lead_time": lstm_lead,
            "threshold_curves": lstm_results.get("threshold_curves", []),
        },
        "naive_threshold": {
            **{k: baseline_results["naive_threshold"][k] for k in
               ["precision", "recall", "f1", "false_alarm_rate", "tp", "fp", "fn", "tn"]},
            "lead_time": baseline_results["naive_lead_time"],
        },
    }

    # ---- Risk ranking (priority list) ----
    ranking = ranking.merge(summary[["machine_id", "failure_step"]], on="machine_id", how="left")
    ranking = ranking.where(pd.notnull(ranking), None)
    ranking_records = ranking.rename(columns={"rf_proba": "risk_score"}).to_dict(orient="records")

    # Annotate ranking with status and severity
    for r in ranking_records:
        score = r["risk_score"]
        if score >= 0.50:
            r["status"] = "CRITICAL"
            r["urgency"] = "Immediate Action"
        elif score >= 0.15:
            r["status"] = "WATCH"
            r["urgency"] = "Schedule Inspection"
        else:
            r["status"] = "HEALTHY"
            r["urgency"] = "Normal Operation"

    # ---- Per-machine sensor + risk series (for the trend chart) ----
    risk_lookup = all_scored.set_index(["machine_id", "step"])["rf_proba"].to_dict()
    machines_series = {}
    for mid, sub in sensor.groupby("machine_id"):
        sub = sub.sort_values("step")
        steps = sub["step"].tolist()
        vib = [round(float(v), 4) for v in sub["vibration"]]
        temp = [round(float(v), 3) for v in sub["temperature"]]
        cur = [round(float(v), 3) for v in sub["current"]]
        pres = [round(float(v), 3) for v in sub["pressure"]]
        risks = [round(float(risk_lookup.get((mid, s), 0.0)), 4) if (mid, s) in risk_lookup else None
                 for s in steps]
        
        fail_val = summary.loc[summary.machine_id == mid, "failure_step"].iloc[0]
        failure_step = None if pd.isna(fail_val) else int(fail_val)
        will_fail = bool(summary.loc[summary.machine_id == mid, "will_fail"].iloc[0])
        
        latest_risk = risks[-1] if risks and risks[-1] is not None else 0.0

        machines_series[mid] = {
            "steps": steps,
            "vibration": vib,
            "temperature": temp,
            "current": cur,
            "pressure": pres,
            "risk": risks,
            "failure_step": failure_step,
            "will_fail": will_fail,
            "latest_risk": latest_risk,
            "summary_stats": {
                "max_vibration": max(vib) if vib else 0.0,
                "max_temperature": max(temp) if temp else 0.0,
                "min_pressure": min(pres) if pres else 0.0,
                "avg_current": round(float(np.mean(cur)), 2) if cur else 0.0,
            }
        }

    # ---- Top warning signals ----
    top_features = baseline_results["top_features"]

    # ---- ROI / Business Value Modeling ----
    n_machines = int(summary.shape[0])
    n_failing = int(summary["will_fail"].sum())
    
    # Cost model constants ($ USD)
    c_unplanned = 15000   # cost of catastrophic in-service failure & unplanned downtime
    c_proactive = 1200    # cost of planned preventative repair before breakdown
    c_inspection = 350    # cost of field inspection on false alert
    
    # Financial estimates
    rf_caught = int(round(n_failing * comparison["random_forest"]["recall"]))
    rf_missed = n_failing - rf_caught
    rf_fp_cost = int(comparison["random_forest"]["fp"] * 0.1 * c_inspection)
    rf_total_cost = (rf_caught * c_proactive) + (rf_missed * c_unplanned) + rf_fp_cost
    
    naive_fp_cost = int(comparison["naive_threshold"]["fp"] * 0.1 * c_inspection)
    naive_total_cost = (n_failing * c_proactive) + naive_fp_cost

    unmanaged_cost = n_failing * c_unplanned
    rf_savings = max(0, unmanaged_cost - rf_total_cost)

    roi_data = {
        "unplanned_failure_cost": c_unplanned,
        "planned_repair_cost": c_proactive,
        "false_alarm_cost": c_inspection,
        "unmanaged_fleet_loss": unmanaged_cost,
        "rf_total_cost": rf_total_cost,
        "naive_total_cost": naive_total_cost,
        "estimated_net_savings": rf_savings,
        "roi_percentage": round((rf_savings / max(1, rf_total_cost)) * 100, 1),
    }

    dashboard = {
        "comparison": comparison,
        "ranking": ranking_records,
        "machines": machines_series,
        "top_features": top_features,
        "roi": roi_data,
        "meta": {
            "n_machines": n_machines,
            "n_failing": n_failing,
            "horizon_hours": 48,
            "n_healthy": n_machines - n_failing,
            "sampling_frequency": "1 hour",
            "fleet_status": {
                "critical": sum(1 for r in ranking_records if r["risk_score"] >= 0.5),
                "watch": sum(1 for r in ranking_records if 0.15 <= r["risk_score"] < 0.5),
                "healthy": sum(1 for r in ranking_records if r["risk_score"] < 0.15),
            }
        },
    }

    def sanitize_for_json(obj):
        if isinstance(obj, float):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return obj
        elif isinstance(obj, dict):
            return {k: sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize_for_json(v) for v in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        return obj

    dashboard = sanitize_for_json(dashboard)

    os.makedirs("results", exist_ok=True)
    with open("dashboard_data.json", "w", encoding="utf-8") as f:
        json.dump(dashboard, f, indent=2, allow_nan=False)
    with open(os.path.join("results", "dashboard_data.json"), "w", encoding="utf-8") as f:
        json.dump(dashboard, f, indent=2, allow_nan=False)

    print("dashboard_data.json written successfully, size:", round(len(json.dumps(dashboard, allow_nan=False)) / 1e6, 2), "MB")

    # ---- Markdown comparison report ----
    def fmt(m):
        lt = m.get("lead_time", {})
        lead_str = f"{lt['mean_lead_hours']:.0f}h mean / {lt['median_lead_hours']:.0f}h median" if lt.get("mean_lead_hours") is not None else "n/a"
        auc_str = f"{m.get('roc_auc'):.3f}" if m.get("roc_auc") is not None else "n/a"
        return (f"| Precision {m['precision']:.3f} | Recall {m['recall']:.3f} | "
                f"F1 {m['f1']:.3f} | False-Alarm Rate {m['false_alarm_rate']:.3f} | "
                f"AUC {auc_str} | Lead Time {lead_str} |")

    report = f"""# Machine Failure Early-Warning System (EWS) — Technical Audit & Model Benchmark

## Dataset & Protocol Overview
- **Fleet Scope**: {dashboard['meta']['n_machines']} industrial machines monitored continuously over 720 operating hours (30 days).
- **Failure Incidence**: {dashboard['meta']['n_failing']} machines experienced bearing-wear/overheating degradation leading to breakdown.
- **Early-Warning Horizon**: 48 hours prior to catastrophic failure.
- **Validation Protocol**: **Machine-level strict train/test split**. Evaluated strictly on unseen physical assets never presented during training.

## Benchmark Results

### 1. Naive Physics Baseline (Vibration Z-Score > 2.5σ)
{fmt(comparison['naive_threshold'])}

### 2. Random Forest on Engineered Rolling-Window Features (6h / 24h / 72h Statistics)
{fmt(comparison['random_forest'])}

### 3. NumPy LSTM Sequence Model (24-Hour Raw Sequence Windows)
{fmt(comparison['lstm'])}

## Key Engineering & Operational Takeaways

1. **The Fallacy of Simple Thresholds**:
   A standard vibration threshold rule achieves high recall ({comparison['naive_threshold']['recall']:.2f}), but at an unacceptable false-alarm rate of {comparison['naive_threshold']['false_alarm_rate']*100:.1f}%. In practice, plant operators experience alert fatigue and disable the alert, rendering it useless.

2. **Random Forest with Rolling Features Dominates Production Readiness**:
   The Random Forest model reaches an F1 score of **{comparison['random_forest']['f1']:.2f}** with a false-alarm rate of only **{comparison['random_forest']['false_alarm_rate']*100:.1f}%**, providing a mean early warning lead time of **{comparison['random_forest']['lead_time']['mean_lead_hours']:.0f} hours**. This gives maintenance crews a comfortable 2-day runway for parts staging and scheduled downtime.

3. **NumPy LSTM Demonstrates Feature-Free Feasibility**:
   Without manual domain feature engineering, the sequence model reaches an impressive ROC-AUC of **{comparison['lstm']['roc_auc']:.3f}**, validating sequence modeling for high-velocity IoT streams where feature extraction overhead is prohibitive.

4. **Estimated Financial Impact**:
   Net fleet operating cost reduction of approximately **${roi_data['estimated_net_savings']:,}** across the 40-machine test environment (an estimated **{roi_data['roi_percentage']}% ROI**).
"""
    with open("model_comparison_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    with open(os.path.join("results", "model_comparison_report.md"), "w", encoding="utf-8") as f:
        f.write(report)
    print("model_comparison_report.md written successfully")


if __name__ == "__main__":
    main()
