"""Alert delivery — sends notifications via Telegram."""

import sys
import json
from urllib.request import Request, urlopen
from urllib.error import URLError


def send_telegram(token, chat_id, message):
    """Send a message via Telegram Bot API. Returns True on success."""
    url = f'https://api.telegram.org/bot{token}/sendMessage'
    payload = json.dumps({'chat_id': chat_id, 'text': message}).encode()
    req = Request(url, data=payload, headers={'Content-Type': 'application/json'})

    try:
        with urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except (URLError, OSError) as e:
        print(f"telegram alert failed: {e}", file=sys.stderr)
        return False
