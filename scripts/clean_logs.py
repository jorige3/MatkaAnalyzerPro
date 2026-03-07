import os
import re

def clean_log(log_path):
    if not os.path.exists(log_path):
        print(f"Log file {log_path} does not exist.")
        return

    with open(log_path, 'r') as f:
        content = f.read()

    entry_marker = "Scraping new data..."
    entries = content.split(entry_marker)
    
    if len(entries) <= 1:
        print("No multiple entries found to clean.")
        return

    preamble = entries[0]
    entry_blocks = entries[1:]

    # Map to store the LATEST entry for each unique "Latest Result" line
    latest_result_map = {}
    
    # regex to find the Latest Result line
    result_pattern = re.compile(r"Latest Result: (.+)")

    for block in entry_blocks:
        match = result_pattern.search(block)
        if match:
            result_key = match.group(1).strip()
            # Store the block, overwriting any previous entry for this result
            # so we always keep the most recent run for a given date/jodi.
            latest_result_map[result_key] = block
        else:
            # If no result found (e.g. error or different output), keep it by hash
            latest_result_map[hash(block)] = block

    # Reconstruct the log
    new_content = preamble
    # Sort keys if possible to maintain some chronological sense, 
    # though dictionary insertion order works for Python 3.7+
    for block in latest_result_map.values():
        new_content += entry_marker + block

    with open(log_path, 'w') as f:
        f.write(new_content)
    
    print(f"Cleaned {len(entry_blocks)} entries down to {len(latest_result_map)} unique results.")

if __name__ == "__main__":
    clean_log("logs/matka_cron.log")
