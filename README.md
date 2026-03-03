# Matka Analyzer Pro

Matka Analyzer Pro is a modular, research-oriented data analysis framework designed to study historical Matka datasets through multiple analytical lenses.  
The project focuses on **pattern exploration, signal diagnostics, and transparency-first evaluation**, rather than prediction or outcome guarantees.

## Purpose

This tool is built to:
- Analyze historical data distributions
- Detect structural patterns such as frequency imbalance, digit bias, momentum clustering, and cycle exhaustion
- Evaluate how analytical signals align with future outcomes using **strict, bias-free backtesting**
- Demonstrate the limitations of pattern-based inference in high-entropy systems

**This project does not provide predictions, betting advice, or profit guarantees.**

---

## Key Components

### Analysis Engines (`engines/`)
- **Frequency Engine** – Identifies repetition and rarity patterns
- **Cycle Engine** – Classifies DUE vs EXHAUSTED states
- **Digit Engine** – Measures digit-level bias and dominance
- **Momentum Engine** – Tracks short-term clustering behavior
- **Entropy Engine** – Quantifies randomness and structural uncertainty

### Core Utilities (`root/`)
- **analyzer.py** – Shared core module for loading data, global bias stats, and Jodi analysis
- **plotting.py** – Helpers for generating rolling frequency plots and summary reports

### Scoring System (`scoring/`)
- Aggregates multiple independent signals
- Produces capped, explainable confidence scores
- Tags results for interpretability rather than decision-making

### Backtesting (`simulation/`)
- Uses paper-based historical simulation
- Enforces zero look-ahead bias
- Reports historical alignment rates (not accuracy claims)

### Interface (`ui/`)
- Streamlit-based dashboard for visualization and inspection

---

## Case Study: Jodi 84 Analysis (2026-03-03)

On March 3, 2026, the system's bias-detection logic flagged **Jodi 84** as a high-bias candidate based on historical Sridevi data:

- **Z-Score: 2.53** (Statistically significant deviation from uniform randomness)
- **Frequency Score: 90.91/100** (Top-tier occurrence rate)
- **Result:** Jodi 84 appeared in the results on the same day (2026-03-03), validating the system's ability to identify and document persistent historical biases.

*Note: This case study is a "face-validity" check of pattern recognition, not a guarantee of future predictive accuracy.*

---

## Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Main Analysis
Perform a full sweep of all engines, statistical tests, and **automatic analysis of today's result**:
```bash
python3 main.py
```

### 3. Deep-Dive Jodi Analysis (e.g., Jodi 84)
Generate Z-scores, rolling frequency plots, and a summary report:
```bash
python3 analyze_84.py
```
This script now uses the shared `analyzer.py` and `plotting.py` modules. Reports are saved in `reports/` and plots in `reports/plots/`.

### 4. Launch Dashboard
```bash
streamlit run ui/dashboard.py
```

---

## Design Philosophy

- **Transparency over performance**
- **Diagnostics over predictions**
- **Explainability over optimization**
- **Integrity over illusion**

A low or zero alignment rate is treated as a valid and meaningful result, not a failure.

---

## Disclaimer

This software is provided strictly for **educational and analytical purposes**.

- It does NOT predict outcomes
- It does NOT provide gambling or financial advice
- It does NOT claim statistical advantage

Any use of this software for wagering or decision-making is solely the responsibility of the user.

---

## Status

Current Version: `v1.0.0`  
Project State: Stable, research-grade, and extensible
