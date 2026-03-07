"""
Kalyan Scraper
--------------
Scrapes Kalyan historical data to enable cross-market analysis.
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import logging
import os
import re

LOG_DIR = 'logs'
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

logging.basicConfig(level=logging.INFO, filename=os.path.join(LOG_DIR, 'scrape_kalyan.log'), filemode='w')
logger = logging.getLogger(__name__)

def scrape_kalyan():
    url = 'https://dpboss.boston/panel-chart-record/kalyan.php#bottom'
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching Kalyan: {e}")
        return []

    soup = BeautifulSoup(resp.text, 'html.parser')
    table = soup.find('table', class_='table-bordered')
    if not table:
        table = soup.find('table')
        if not table:
            return []

    historical_data = []
    date_pattern = re.compile(r'(\d{2}/\d{2}/\d{4})')
    rows = table.find_all('tr')

    for row in rows:
        cells = row.find_all('td')
        if len(cells) < 2:
            continue
        base_date = None
        date_cell_index = -1
        for i, cell in enumerate(cells):
            match = date_pattern.search(cell.get_text())
            if match:
                try:
                    base_date = datetime.strptime(match.group(1), '%d/%m/%Y')
                    date_cell_index = i
                    break
                except ValueError:
                    continue
        if not base_date:
            continue
        day_offset = 0
        for i in range(date_cell_index, len(cells)):
            cell = cells[i]
            text = cell.get_text().strip()
            if text.isdigit() and len(text) == 2:
                current_date = base_date + timedelta(days=day_offset)
                historical_data.append({'Date': current_date.strftime('%Y-%m-%d'), 'Jodi': text})
                day_offset += 1
            elif "***" in text or "**" in text or not text:
                if i > date_cell_index:
                    day_offset += 1
    return historical_data

if __name__ == '__main__':
    data = scrape_kalyan()
    if data:
        df = pd.DataFrame(data)
        df.to_csv('data/kalyan.csv', index=False)
        print(f"Scraped {len(df)} Kalyan records.")
