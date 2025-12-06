#!/usr/bin/env python3
"""Ingest NFL game data from nflverse using nflreadpy."""

import argparse
from pathlib import Path
import sys

import nflreadpy as nfl

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nfl_predictor.data.ingest_games import (
    normalize_nflverse_schedule,
    print_summary_stats,
)


def main():
    """Fetch and save NFL games from nflverse."""
    parser = argparse.ArgumentParser(
        description="Ingest NFL game data from nflverse"
    )
    parser.add_argument(
        "--start-season",
        type=int,
        default=2000,
        help="First season to fetch (default: 2000)",
    )
    parser.add_argument(
        "--end-season",
        type=int,
        default=2024,
        help="Last season to fetch (default: 2024)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/interim/games_base.parquet"),
        help="Output path for parquet file (default: data/interim/games_base.parquet)",
    )

    args = parser.parse_args()

    # Create output directory if needed
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Fetch schedule data
    print(f"\nFetching NFL schedules for seasons {args.start_season}-{args.end_season}...")
    seasons = list(range(args.start_season, args.end_season + 1))

    try:
        df_raw = nfl.load_schedules(seasons)
        print(f"Fetched {len(df_raw):,} total schedule records")
    except Exception as e:
        print(f"Error fetching data: {e}")
        sys.exit(1)

    # Normalize to GameBase schema
    print("\nNormalizing to GameBase schema...")
    try:
        df_normalized = normalize_nflverse_schedule(df_raw)
        print(f"Normalized to {len(df_normalized):,} completed games")
    except Exception as e:
        print(f"Error normalizing data: {e}")
        sys.exit(1)

    # Save to parquet
    print(f"\nSaving to {args.output}...")
    df_normalized.to_parquet(args.output, index=False)
    print(f"Saved {len(df_normalized):,} games to {args.output}")

    # Print summary stats
    print_summary_stats(df_normalized)

    print("\nIngestion complete!")


if __name__ == "__main__":
    main()
