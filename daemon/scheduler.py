"""Simple scheduler — runs functions at fixed intervals."""

import time


class Scheduler:
    """Minimal scheduler using time.sleep(). No threading."""

    def __init__(self):
        self.jobs = []

    def every(self, minutes, fn, name=None):
        """Schedule fn to run every N minutes."""
        self.jobs.append({
            'fn': fn,
            'interval': minutes * 60,
            'name': name or fn.__name__,
            'last_run': 0,
        })
        return self

    def run_forever(self):
        """Block and run scheduled jobs. Ctrl+C to stop."""
        print(f"scheduler started with {len(self.jobs)} job(s)")
        try:
            while True:
                now = time.time()
                for job in self.jobs:
                    if now - job['last_run'] >= job['interval']:
                        try:
                            job['fn']()
                        except Exception as e:
                            print(f"job '{job['name']}' failed: {e}")
                        job['last_run'] = time.time()
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nscheduler stopped")
