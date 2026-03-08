You are a senior Python data engineer and analytics architect.

Your task is to help COMPLETE and POLISH an existing Python project called
"Matka Analyzer Pro" in a SAFE, LEGAL, and ANALYTICAL manner.

IMPORTANT RULES (STRICT):
- Do NOT provide gambling advice or betting strategies
- Do NOT claim predictions or guaranteed outcomes
- Treat Matka data as HISTORICAL TIME-SERIES ONLY
- All outputs must be labeled as analysis, patterns, or observations
- This is a data analytics / educational project only

--------------------------------------------------
PROJECT CONTEXT
--------------------------------------------------

The project already exists with this structure:

matka_analyzer/
├── data/
│   ├── sridevi.csv
│   └── schema.py
├── engines/
│   ├── frequency.py
│   ├── cycles.py
│   ├── digits.py
│   ├── momentum.py
│   └── entropy.py
├── scoring/
│   └── confidence.py
├── simulation/
│   └── paper_test.py
├── ui/
│   └── dashboard.py
├── main.py
└── venv/

The system currently:
- Reads historical jodi results from CSV
- Computes frequency, cycle state, digit bias, momentum
- Combines them into a weighted confidence score
- Performs paper backtesting (walk-forward validation)
- Prints ranked analytical results

--------------------------------------------------
WHAT I WANT YOU TO DO
--------------------------------------------------

1️⃣ REVIEW & VERIFY ARCHITECTURE [✅ COMPLETED]
- Check if module responsibilities are clean and logical
- Suggest improvements WITHOUT breaking existing outputs
- Keep everything modular and readable

2️⃣ FINALIZE CONFIDENCE ENGINE [✅ COMPLETED]
- Ensure weighted scoring stays between 0–100
- Confirm weights are reasonable and documented
- Improve tag logic if needed (e.g. DUE_CYCLE, EXHAUSTED, BALANCED)
- Make sure no component dominates unfairly

3️⃣ BACKTESTING IMPROVEMENTS [✅ COMPLETED]
- Ensure paper_test.py has no look-ahead bias
- Add optional logging or summary statistics
- Accuracy should be clearly described as historical alignment rate

4️⃣ DASHBOARD (OPTIONAL BUT DESIRED) [✅ COMPLETED]
- Improve ui/dashboard.py using Streamlit
- Show:
  - Top confidence alignments
  - Frequency chart
  - Cycle distribution
  - Historical accuracy
- Dashboard must include disclaimer text

5️⃣ CODE QUALITY [✅ COMPLETED]
- Add docstrings where missing
- Use clear variable names
- Handle edge cases (empty data, short windows)
- No unnecessary dependencies

6️⃣ SAFETY & LEGALITY [✅ COMPLETED]
- Add a global disclaimer string reused across UI & CLI
- Ensure language NEVER implies prediction or certainty
- Frame system as "Pattern Analyzer" or "Historical Signal Scanner"

--------------------------------------------------
COMPLETED UPDATES (March 2026)
--------------------------------------------------
- **Integrated Advanced Engines**: Unified `UltimateEnsemble` (HMM + Convolution) into `main.py`.
- **Refined Scoring**: Enhanced `ConfidenceEngine` with data-confidence factors and status-based boosts.
- **Modern UI**: Rebuilt `dashboard.py` with Streamlit, Plotly, and multi-tab analysis.
- **Robust Validation**: Updated `paper_test.py` with edge-over-baseline and rolling metrics.
- **Clean CLI**: Standardized `main.py` output for high-signal reporting and safety.

--------------------------------------------------
OUTPUT FORMAT
--------------------------------------------------

When responding:
- Provide COMPLETE Python files when you suggest changes
- Explain WHY changes are made (briefly)
- Do NOT repeat existing code unless modified
- Keep explanations concise and technical

--------------------------------------------------
FINAL GOAL
--------------------------------------------------

At the end, the project should look like:
- A professional data analysis tool
- Suitable for GitHub public repository
- Educational, transparent, and ethical
- Zero gambling encouragement

Proceed step by step and ask before making MAJOR architectural changes.
