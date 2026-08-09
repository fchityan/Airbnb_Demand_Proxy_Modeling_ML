from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path


def cleanup_expired_runs(output_dir: Path, retention_days: int = 30) -> list[str]:
    runs_dir = output_dir / "runs"
    if not runs_dir.exists():
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    deleted: list[str] = []

    for child in runs_dir.iterdir():
        if not child.is_dir():
            continue
        modified = datetime.fromtimestamp(child.stat().st_mtime, tz=timezone.utc)
        if modified < cutoff:
            shutil.rmtree(child)
            deleted.append(child.name)

    return deleted


def main() -> None:
    parser = argparse.ArgumentParser(description="Operational helpers for ML pipeline artifacts.")
    parser.add_argument("--output-dir", default="outputs", help="Root output directory containing run artifacts.")
    parser.add_argument("--retention-days", type=int, default=30, help="Retention window in days.")
    args = parser.parse_args()

    deleted = cleanup_expired_runs(Path(args.output_dir), retention_days=args.retention_days)
    print(f"Deleted {len(deleted)} expired run directories.")


if __name__ == "__main__":
    main()
