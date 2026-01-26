"""
Background log streamer for Koyeb deployments.

Reads run.log periodically and POSTs contents to the koyeb-monitor app.
Replaces the fragile inline `python -c` one-liner in deploy_koyeb.sh.

Usage (from deploy_koyeb.sh):
    python utils/log_streamer.py &
    LOG_PID=$!

Environment variables (required):
    MONITOR_URL: Base URL of the koyeb-monitor app
    KOYEB_APP_NAME: Name of the Koyeb app (used as log key)

Environment variables (optional):
    LOG_FILE: Path to log file (default: run.log)
    LOG_INTERVAL: Seconds between submissions (default: 60)
"""

import os
import sys
import time
import requests

MONITOR_URL = os.environ.get("MONITOR_URL", "")
APP_NAME = os.environ.get("KOYEB_APP_NAME", "")
LOG_FILE = os.environ.get("LOG_FILE", "run.log")
INTERVAL = int(os.environ.get("LOG_INTERVAL", "60"))


def submit_logs():
    """Read log file and POST to monitor."""
    if os.path.exists(LOG_FILE):
        logs = open(LOG_FILE).read()
    else:
        logs = "(log file not yet created)"

    resp = requests.post(
        f"{MONITOR_URL}/submit-logs",
        json={"app_name": APP_NAME, "logs": logs},
        timeout=30,
    )
    resp.raise_for_status()


def main():
    if not MONITOR_URL or not APP_NAME:
        print("log_streamer: MONITOR_URL and KOYEB_APP_NAME must be set", file=sys.stderr)
        sys.exit(1)

    print(f"log_streamer: streaming {LOG_FILE} to {MONITOR_URL} every {INTERVAL}s", file=sys.stderr)

    while True:
        time.sleep(INTERVAL)
        try:
            submit_logs()
        except Exception as e:
            print(f"log_streamer: error: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
