"""CLI entry point.

    python -m scripts.run_pipeline                     # default config
    python -m scripts.run_pipeline --config config/experiment.yaml
"""
from __future__ import annotations

import argparse

from src.pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Flight delay prediction pipeline")
    parser.add_argument("--config", default=None, help="Path to a YAML config file")
    args = parser.parse_args()
    result = run(args.config)
    print("\nChampion model:", result["champion"])
    print(result["leaderboard"].to_string(index=False))


if __name__ == "__main__":
    main()
