"""KIRI daemon entry point — wires the scheduler to the pulse collector."""

from .config import DATA_DIR, COLLECT_INTERVAL
from .atoms.pulse.collect import collect_local, save_observations
from .daemon.scheduler import Scheduler


def collect_and_save():
    """Single collection cycle."""
    obs = collect_local()
    if obs is None:
        return
    save_observations([obs], str(DATA_DIR))
    print(f"collected observation: C={obs['C']:.0f}% M={obs['M']:.0f}% D={obs['D']:.0f}%")


def main():
    print(f"kiri daemon starting")
    print(f"collecting every {COLLECT_INTERVAL} minutes -> {DATA_DIR}")

    scheduler = Scheduler()
    scheduler.every(COLLECT_INTERVAL, collect_and_save, name='pulse-collect')
    scheduler.run_forever()


if __name__ == '__main__':
    main()
