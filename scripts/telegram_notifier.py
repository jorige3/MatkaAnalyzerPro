import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

class TelegramNotifier:
    """Handles sending notifications to Telegram."""
    
    def __init__(self):
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"

    def send_message(self, message: str):
        """Sends a simple text message."""
        if not self.bot_token or not self.chat_id:
            print("[!] Telegram Credentials not set in .env")
            return False
            
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "Markdown"
        }
        
        try:
            response = requests.post(self.base_url, json=payload)
            response.raise_for_status()
            return True
        except Exception as e:
            print(f"[!] Failed to send Telegram message: {e}")
            return False

if __name__ == "__main__":
    # Quick test
    notifier = TelegramNotifier()
    notifier.send_message("*Test Message* from Matka Analyzer Pro")
