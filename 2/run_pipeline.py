# -*- coding: utf-8 -*-
"""
Master End-to-End Pipeline Runner for Machine Failure Early-Warning System (EWS).

Executes all 6 stages sequentially with benchmarking, health checks, and logging:
  1. Data Simulation (data_gen.py)
  2. Rolling Feature Engineering (features.py)
  3. Baseline Classifier and Lead-Time Analysis (baseline_model.py)
  4. Sequence Deep Learning (lstm_numpy.py)
  5. Dashboard Data Aggregation and ROI Analysis (build_dashboard_data.py)
  6. Self-Contained Dashboard HTML Compilation (build_html.py)
"""
import time
import os
import sys

def banner(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def main():
    total_start = time.time()
    banner("MACHINE FAILURE EARLY-WARNING SYSTEM: PIPELINE EXECUTION")

    stages = [
        ("1/6: Synthetic Sensor Simulation", "data_gen"),
        ("2/6: Rolling-Window Feature Engineering", "features"),
        ("3/6: Random Forest & Physics Baseline Training", "baseline_model"),
        ("4/6: From-Scratch NumPy LSTM Sequence Modeling", "lstm_numpy"),
        ("5/6: Dashboard Data & ROI Aggregation", "build_dashboard_data"),
        ("6/6: Interactive Standalone HTML Compilation", "build_html"),
    ]

    for label, module_name in stages:
        print(f"\n>>> Running Stage {label}...")
        start = time.time()
        try:
            if module_name == "data_gen":
                import data_gen
                data_gen.main()
            elif module_name == "features":
                import features
                features.build_feature_table()
            elif module_name == "baseline_model":
                import baseline_model
                baseline_model.main()
            elif module_name == "lstm_numpy":
                import lstm_numpy
                lstm_numpy.main()
            elif module_name == "build_dashboard_data":
                import build_dashboard_data
                build_dashboard_data.main()
            elif module_name == "build_html":
                import build_html
                build_html.build_dashboard()
        except Exception as e:
            print(f"\n[ERROR] Pipeline failed at stage '{module_name}': {e}")
            sys.exit(1)
        elapsed = time.time() - start
        print(f"    [OK] {module_name} finished in {elapsed:.2f}s")

    total_time = time.time() - total_start
    banner("PIPELINE COMPLETE - READY FOR PRESENTATION")
    print(f"Total execution time: {total_time:.2f} seconds")
    dash_path = os.path.abspath("machine_failure_dashboard.html")
    print(f"Dashboard generated at: {dash_path}")
    print("\nTo view the dashboard:")
    print("  1. Double click 'machine_failure_dashboard.html' in File Explorer, OR")
    print("  2. Run 'python serve.py' to launch the presentation server on http://localhost:8000")

if __name__ == '__main__':
    main()
