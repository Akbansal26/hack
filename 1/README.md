# ⚙️ Machine Failure Early-Warning System (EWS)
> **An Industrial IoT Predictive Maintenance Intelligence Suite**
> *Evaluating classical rolling-window feature engineering (Random Forest) vs. sequence deep learning (NumPy LSTM) vs. physics thresholds on held-out physical assets.*

---

## 🌟 Overview & Hackathon Highlights

Unplanned equipment downtime costs industrial manufacturers an estimated **$50 Billion annually**. Traditional monitoring systems rely on simple threshold alarms (e.g. vibration > 2.5σ), which suffer from a debilitating **46.9% false-alarm rate**, leading to severe operator alert fatigue.

This project delivers an end-to-end, production-ready **Early-Warning System (EWS)** that flags impending bearing wear, overheating, and pressure drop **~50 hours in advance**, while suppressing false alarms down to **1.1%** — translating to a **~75% operational cost reduction** ($230,000+ estimated net savings across 40 monitored assets).

### 🚀 Key Innovations
1. **Strict Machine-Level Generalization**:
   Unlike flawed row-level cross-validation that leaks asset identity, the evaluation splits by entire machine IDs (`test_machines` never seen during training).
2. **Multi-Paradigm Benchmark**:
   - **Physics Rule**: Baseline vibration z-score (shows the fallacy of raw thresholds).
   - **Feature-Engineered Random Forest**: 6h/24h/72h rolling statistics + differential velocity features (**F1: 0.78, ROC-AUC: 0.996**).
   - **From-Scratch NumPy LSTM**: Full BPTT + Adam optimizer in pure NumPy without PyTorch/TensorFlow dependencies (**ROC-AUC: 0.986** on raw waveforms).
3. **Executive Presentation Dashboard**:
   - 100% self-contained, zero-dependency, offline-resilient HTML dashboard (`machine_failure_dashboard.html`).
   - Interactive **Alert Threshold Slider (0.10 - 0.90)** demonstrating live precision/recall trade-offs.
   - **Live Telemetry Stream Simulator** with real-time risk gauges.
   - Fleet Priority Matrix with critical/watch/healthy filtering.
   - Economic ROI impact modeling.

---

## 📊 Benchmark Results Summary

| Model Architecture | Precision | Recall | F1-Score | False-Alarm Rate | ROC-AUC | Mean Early Warning Runway |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naive Physics Rule** (Vib > 2.5σ) | 0.066 | **0.983** | 0.124 | 46.89% | — | 294h *(unreliable)* |
| **⭐ Random Forest** (Rolling Features) | **0.724** | 0.842 | **0.778** | **1.08%** | **0.996** | **50.0 Hours** |
| **NumPy LSTM** (Raw Sequences) | 0.623 | 0.867 | 0.725 | 1.77% | 0.986 | 53.0 Hours |

---

## ⚡ Quickstart & How to Run

### 1. Requirements
```bash
pip install pandas numpy scikit-learn
```

### 2. One-Click Full Pipeline Execution
Run all 6 stages (data simulation, feature extraction, RF training, LSTM training, dashboard data compilation, HTML build):
```bash
python run_pipeline.py
```

### 3. Launch Presentation Server
```bash
python serve.py
# Opens http://localhost:8000/machine_failure_dashboard.html automatically in your default browser
```
*Alternatively, simply double-click `machine_failure_dashboard.html` in File Explorer — it runs 100% offline via `file:///` without any web server needed!*

### 4. Run Automated Verification Test Suite
```bash
python test_pipeline.py
```

---

## 📁 Repository Structure

```
project_export/
├── run_pipeline.py              # Master 1-click end-to-end pipeline runner
├── serve.py                     # Local presentation HTTP server (auto-opens browser)
├── test_pipeline.py             # 7-point automated verification & test suite
├── data_gen.py                  # Realistic 4-sensor degradation & failure simulation
├── features.py                  # Rolling-window statistical feature engineering
├── baseline_model.py            # Machine-split Random Forest & threshold curves
├── lstm_numpy.py                # Pure NumPy sequence LSTM (forward + BPTT + Adam)
├── build_dashboard_data.py      # Aggregates metrics, ROI, and JSON payload
├── build_html.py                # Compiles self-contained interactive dashboard
├── machine_failure_dashboard.html # Production interactive dashboard (HTML5 + Chart.js)
├── chart.umd.min.js             # Bundled local offline Chart.js library
├── model_comparison_report.md   # Detailed audit report & takeaways
├── data/                        # Mirrored synthetic sensor & machine datasets
│   ├── sensor_data.csv
│   ├── machine_summary.csv
│   └── features.csv
└── results/                     # Benchmark artifacts & predictions
    ├── baseline_results.json
    ├── lstm_results.json
    ├── machine_risk_ranking.csv
    └── test_scored.csv
```

---

## 🎯 Hackathon Presentation Demo Flow

1. **The Hook (Executive KPIs)**:
   Point to the **1.1% vs. 46.9% False-Alarm Rate** and the **$231,905 Net Savings**. Explain that raw thresholds cause alert fatigue, whereas EWS provides an actionable 50-hour runway.
2. **Interactive Threshold Tuning**:
   Move the slider on the dashboard from `0.50` down to `0.20` or up to `0.80`. Show the judges how operations can tune sensitivity based on critical vs. non-critical assets.
3. **Live Telemetry Stream Simulator**:
   Click **"▶ Live Stream Demo"** on unit `M037`. Watch the hours advance, the sensor waveforms ramp up, and the risk gauge trigger a critical alert exactly within the 48-hour degradation runway before failure!
4. **Offline Resilience**:
   Demonstrate that the entire dashboard is self-contained and loads instantly even without internet access.

