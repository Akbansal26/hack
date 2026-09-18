# -*- coding: utf-8 -*-
"""
Automated end-to-end test suite for Machine Failure Early-Warning System (EWS).
Validates data integrity, model metric bounds, JSON compliance, and HTML rendering.
"""
import unittest
import os
import json
import pandas as pd
import numpy as np

class TestEarlyWarningSystem(unittest.TestCase):
    def setUp(self):
        self.project_dir = os.path.abspath(os.path.dirname(__file__))

    def test_01_sensor_data_integrity(self):
        csv_path = os.path.join(self.project_dir, "sensor_data.csv")
        self.assertTrue(os.path.exists(csv_path), "sensor_data.csv missing")
        df = pd.read_csv(csv_path)
        self.assertGreater(len(df), 15000, "sensor data too small")
        for col in ["machine_id", "timestamp", "step", "vibration", "temperature", "current", "pressure", "failure_event"]:
            self.assertIn(col, df.columns, f"missing column {col}")
        self.assertEqual(df["vibration"].isnull().sum(), 0, "NaN found in vibration")
        self.assertEqual(df["temperature"].isnull().sum(), 0, "NaN found in temperature")
        self.assertEqual(df["current"].isnull().sum(), 0, "NaN found in current")
        self.assertEqual(df["pressure"].isnull().sum(), 0, "NaN found in pressure")

    def test_02_machine_summary(self):
        csv_path = os.path.join(self.project_dir, "machine_summary.csv")
        self.assertTrue(os.path.exists(csv_path), "machine_summary.csv missing")
        df = pd.read_csv(csv_path)
        self.assertEqual(len(df), 40, "expected 40 machines")
        self.assertGreater(df["will_fail"].sum(), 10, "insufficient failing machines")

    def test_03_features_table(self):
        feat_path = os.path.join(self.project_dir, "features.csv")
        self.assertTrue(os.path.exists(feat_path), "features.csv missing")
        df = pd.read_csv(feat_path)
        self.assertGreater(len(df), 15000, "features table too small")
        self.assertIn("label", df.columns, "target label column missing")
        pos_rate = df["label"].mean()
        self.assertTrue(0.01 <= pos_rate <= 0.15, f"unreasonable positive rate: {pos_rate}")

    def test_04_baseline_results(self):
        json_path = os.path.join(self.project_dir, "baseline_results.json")
        self.assertTrue(os.path.exists(json_path), "baseline_results.json missing")
        with open(json_path, "r", encoding="utf-8") as f:
            res = json.load(f)
        rf = res["random_forest"]
        self.assertGreaterEqual(rf["f1"], 0.70, f"Random Forest F1 below 0.70: {rf['f1']}")
        self.assertLessEqual(rf["false_alarm_rate"], 0.05, f"False alarm rate above 5%: {rf['false_alarm_rate']}")
        lead = res["rf_lead_time"]
        self.assertGreaterEqual(lead["mean_lead_hours"], 30.0, f"Mean lead time below 30h: {lead['mean_lead_hours']}")
        self.assertEqual(lead["n_missed"], 0, "Random Forest missed failing test machines")

    def test_05_lstm_results(self):
        json_path = os.path.join(self.project_dir, "lstm_results.json")
        self.assertTrue(os.path.exists(json_path), "lstm_results.json missing")
        with open(json_path, "r", encoding="utf-8") as f:
            res = json.load(f)
        self.assertGreaterEqual(res["roc_auc"], 0.90, f"LSTM ROC AUC below 0.90: {res['roc_auc']}")

    def test_06_dashboard_json_compliance(self):
        json_path = os.path.join(self.project_dir, "dashboard_data.json")
        self.assertTrue(os.path.exists(json_path), "dashboard_data.json missing")
        with open(json_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        self.assertNotIn("NaN", raw_text, "Found unquoted NaN in dashboard_data.json!")
        self.assertNotIn("Infinity", raw_text, "Found unquoted Infinity in dashboard_data.json!")
        data = json.loads(raw_text)
        self.assertIn("comparison", data)
        self.assertIn("ranking", data)
        self.assertIn("machines", data)
        self.assertIn("roi", data)
        self.assertEqual(len(data["machines"]), 40, "Expected 40 machine series in dashboard data")

    def test_07_dashboard_html_resilience(self):
        html_path = os.path.join(self.project_dir, "machine_failure_dashboard.html")
        self.assertTrue(os.path.exists(html_path), "machine_failure_dashboard.html missing")
        with open(html_path, "r", encoding="utf-8") as f:
            html = f.read()
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("Chart.js", html)
        self.assertIn("Early-Warning System", html)
        self.assertIn("dashboard-data", html)
        self.assertIn("report-md", html)
        self.assertGreater(len(html), 1000000, "HTML appears incomplete or unbundled")

if __name__ == "__main__":
    unittest.main(verbosity=2)
