#!/bin/bash

# Navigate to the project directory
cd /home/kishore/MatkaAnalyzerPro/matka_analyzer

# Activate the virtual environment
source venv/bin/activate

# 1. Scrape new data
echo "Scraping new data..."
python scripts/scrape_sridevi.py

# 2. Run the main analysis and ML prediction
echo "Running main analysis and ML prediction..."
# Ensure PYTHONPATH is set for main.py to find modules
PYTHONPATH=. python main.py

# 3. Git operations
echo "Committing changes..."
git add data/sridevi.csv
git add .
git commit -m "feat: Daily data update and analysis run"

echo "Daily automation complete."

# 4. Clean up logs to remove redundant entries
echo "Cleaning up redundant log entries..."
PYTHONPATH=. python scripts/clean_logs.py
