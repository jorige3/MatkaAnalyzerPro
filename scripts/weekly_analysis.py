"""
Weekly Analysis & Telegram Notification
----------------------------------------
Runs analysis for both Sridevi and Kalyan markets and sends a summary to Telegram.
"""

import os
import sys
import pandas as pd
from datetime import datetime

# Add project root to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from main import run_classic_engines
from data.data_loader import DataLoader
from scripts.telegram_notifier import TelegramNotifier
from config import MARKETS, SCHEMA_FILE, DISCLAIMER, TOP_N_PREDICTIONS

def format_market_summary(market_name: str, results: dict, top_k: int = 3) -> str:
    """Formats the results for a single market into a concise Telegram string."""
    summary = f"📊 *{market_name.upper()} ANALYSIS*\n"
    summary += f"Rank | Jodi | Score | Tags\n"
    summary += f"-----|------|-------|------\n"
    
    for i, (jodi, score, tags) in enumerate(results["confidence"][:top_k], 1):
        tag_str = ", ".join(tags[:2]) # Max 2 tags to keep it concise
        summary += f"{i}. *{jodi}* | {score:.1f} | {tag_str}\n"
    
    return summary

def main():
    print(f"[*] Starting Weekly Analysis on {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    full_report = "🚀 *MATKA ANALYZER PRO: WEEKLY SUMMARY*\n"
    full_report += f"Date: {datetime.now().strftime('%A, %b %d, %Y')}\n\n"
    
    for market_name, data_path in MARKETS.items():
        print(f"[*] Processing {market_name}...")
        try:
            loader = DataLoader(data_path, SCHEMA_FILE)
            df = loader.load_data()
            
            # Run classic engines
            results = run_classic_engines(df)
            
            # Format and append
            full_report += format_market_summary(market_name, results)
            full_report += "\n"
            
        except Exception as e:
            print(f"[!] Error processing {market_name}: {e}")
            full_report += f"⚠️ *{market_name}*: Error during analysis.\n\n"

    full_report += f"---\n_{DISCLAIMER}_"
    
    # Send via Telegram
    print("[*] Sending Telegram Notification...")
    notifier = TelegramNotifier()
    success = notifier.send_message(full_report)
    
    if success:
        print("[+] Weekly report sent successfully!")
    else:
        print("[!] Failed to send weekly report.")

if __name__ == "__main__":
    main()
