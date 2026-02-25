# scripts/scrape_sridevi.py
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import logging
import os
import re

# Create logs directory if it doesn't exist
LOG_DIR = 'logs'
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

logging.basicConfig(level=logging.INFO, filename=os.path.join(LOG_DIR, 'scrape.log'), filemode='w') # Overwrite log each time
logger = logging.getLogger(__name__)

def scrape_historical():
    """
    Scrapes the entire historical data table from the website,
    handling weekly data layout.
    """
    url = 'https://dpboss.boston/panel-chart-record/sridevi.php#bottom'
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching URL: {e}")
        return []

    soup = BeautifulSoup(resp.text, 'html.parser')
    
    table = soup.find('table', class_='table-bordered')
    if not table:
        logger.warning("Could not find a table with class 'table-bordered'. Trying to find any table.")
        table = soup.find('table')
        if not table:
            logger.error("No table found on the page.")
            return []

    historical_data = []
    date_pattern = re.compile(r'(\d{2}/\d{2}/\d{4})')

    rows = table.find_all('tr')
    logger.info(f"Found {len(rows)} rows in the table.")

    for row in rows:
        cells = row.find_all('td')
        if len(cells) < 2:  # Need at least a date and one jodi
            continue

        base_date = None
        date_cell_index = -1

        # Find the starting date in the row
        for i, cell in enumerate(cells):
            match = date_pattern.search(cell.get_text())
            if match:
                try:
                    base_date = datetime.strptime(match.group(1), '%d/%m/%Y')
                    date_cell_index = i
                    break
                except ValueError as e:
                    logger.error(f"Error parsing date '{match.group(1)}': {e}")
                    continue
        
        if not base_date:
            continue

        # Iterate through cells again to extract jodis, using date cell as anchor
        day_offset = 0
        for i in range(date_cell_index, len(cells)):
            cell = cells[i]
            text = cell.get_text().strip()
            if text.isdigit() and len(text) == 2:
                current_date = base_date + timedelta(days=day_offset)
                formatted_date = current_date.strftime('%Y-%m-%d')
                historical_data.append({'Date': formatted_date, 'Jodi': text})
                day_offset += 1
            # Handle cases where a cell might contain non-jodi text but still occupies a day slot
            elif i > date_cell_index:
                 # This simple logic assumes every cell after the date is a new day
                 # A more robust solution would be needed if the table layout is inconsistent
                 pass


    logger.info(f"Scraped {len(historical_data)} historical records.")
    return historical_data

def append_to_csv(new_data, csv_path='data/sridevi.csv'):
    """
    Appends a list of new data to the CSV, checking for duplicates.
    """
    if not new_data:
        logger.info("No new data to append.")
        return

    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        try:
            df_existing = pd.read_csv(csv_path)
            # Create a set of existing dates for faster lookup
            existing_dates = set(df_existing['Date'])
            # Filter out data that already exists
            new_data_to_add = [row for row in new_data if row['Date'] not in existing_dates]
            
            if not new_data_to_add:
                logger.info("All scraped data already exists in the CSV. No new data to append.")
                return
            
            df_new = pd.DataFrame(new_data_to_add)

        except (pd.errors.EmptyDataError, KeyError) as e:
            logger.warning(f"Could not read existing CSV or it's malformed: {e}. Overwriting with new data.")
            df_new = pd.DataFrame(new_data)
    else:
        df_new = pd.DataFrame(new_data)
    
    # Append or write new data
    mode = 'a' if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0 else 'w'
    header = not (os.path.exists(csv_path) and os.path.getsize(csv_path) > 0)
    df_new.to_csv(csv_path, mode=mode, header=header, index=False)
    logger.info(f"Appended {len(df_new)} new rows to {csv_path}")

def overwrite_csv(data, csv_path='data/sridevi.csv'):
    """
    Overwrites the CSV with new data.
    """
    if not data:
        logger.warning(f"No data provided to overwrite {csv_path}. The file will be empty.")
        # Create an empty file with header
        with open(csv_path, 'w') as f:
            f.write('Date,Jodi\n')
        return

    df = pd.DataFrame(data)
    # Ensure columns are in the correct order
    df = df[['Date', 'Jodi']]
    df.to_csv(csv_path, index=False)
    logger.info(f"Overwrote {csv_path} with {len(df)} rows.")

if __name__ == '__main__':
    start_date = '2023-01-01'
    end_date = '2026-12-31'

    # Scrape all historical data
    all_data = scrape_historical()

    # Filter data for the desired date range
    filtered_data = [
        row for row in all_data
        if start_date <= row['Date'] <= end_date
    ]

    # Manually add the most recent data that the historical scraper misses
    recent_data = [
        {'Date': '2026-02-17', 'Jodi': '07'},
        {'Date': '2026-02-18', 'Jodi': '09'},
    ]

    # Add the recent data if it's within the date range
    for row in recent_data:
        if start_date <= row['Date'] <= end_date:
            # Avoid duplicates
            if row['Date'] not in {d['Date'] for d in filtered_data}:
                filtered_data.append(row)

    # Sort data by date
    if filtered_data:
        filtered_data.sort(key=lambda x: x['Date'])

    logger.info(f"Filtered data to {len(filtered_data)} records between {start_date} and {end_date}.")
    if filtered_data:
        logger.info(f"First 5 filtered records:\n{filtered_data[:5]}")
        logger.info(f"Last 5 filtered records:\n{filtered_data[-5:]}")

    # Overwrite the CSV with the filtered data
    overwrite_csv(filtered_data, csv_path='data/sridevi.csv')
