"""Global KIRI configuration. Override via environment variables."""

import os
from pathlib import Path

# Telegram alerts
TELEGRAM_TOKEN = os.environ.get('KIRI_TELEGRAM_TOKEN', '')
TELEGRAM_CHAT_ID = os.environ.get('KIRI_TELEGRAM_CHAT_ID', '')

# Paths
DATA_DIR = Path(__file__).resolve().parent / 'data'

# Collection interval (minutes)
COLLECT_INTERVAL = 5
