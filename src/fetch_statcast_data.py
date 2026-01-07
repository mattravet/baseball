"""
Fetch Statcast batted ball data for marine layer analysis.
"""

import pandas as pd
from pybaseball import statcast
from datetime import datetime
import os

# Statcast data directory
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

# Columns we need for analysis
COLUMNS_TO_KEEP = [
    'game_date',
    'game_pk',
    'at_bat_number',
    'pitch_number',
    'inning',
    'inning_topbot',
    'home_team',
    'away_team',
    'batter',
    'pitcher',
    'events',
    'description',
    'launch_speed',      # exit velocity
    'launch_angle',      # vertical launch angle
    'hit_distance_sc',   # statcast calculated distance
    'hc_x',              # hit coordinate x (for spray angle)
    'hc_y',              # hit coordinate y (for spray angle)
    'bb_type',           # batted ball type (fly_ball, line_drive, ground_ball, popup)
    'estimated_ba_using_speedangle',
    'estimated_woba_using_speedangle',
    'barrel',
]


def fetch_season_data(year: int) -> pd.DataFrame:
    """Fetch Statcast data for a full season."""
    print(f"Fetching {year} season data...")

    # MLB season typically runs April through October
    start_date = f"{year}-03-28"
    end_date = f"{year}-10-05"

    df = statcast(start_dt=start_date, end_dt=end_date)
    print(f"  Raw data: {len(df)} pitches")

    return df


def filter_batted_balls(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to only batted balls with valid tracking data."""
    # Filter to batted balls (balls put in play)
    batted = df[df['description'].isin(['hit_into_play', 'hit_into_play_no_out', 'hit_into_play_score'])].copy()

    # Require valid exit velocity, launch angle, and distance
    batted = batted.dropna(subset=['launch_speed', 'launch_angle', 'hit_distance_sc'])

    # Require valid hit coordinates for spray angle
    batted = batted.dropna(subset=['hc_x', 'hc_y'])

    # Filter out unrealistic values
    batted = batted[
        (batted['launch_speed'] >= 50) &  # Minimum EV of 50 mph
        (batted['launch_speed'] <= 125) &  # Maximum EV of 125 mph
        (batted['hit_distance_sc'] >= 50) &  # Minimum distance 50 feet
        (batted['hit_distance_sc'] <= 500)   # Maximum distance 500 feet
    ]

    print(f"  Filtered batted balls: {len(batted)}")

    return batted


def process_and_save(years: list[int]):
    """Fetch, process, and save data for multiple years."""
    all_data = []

    for year in years:
        df = fetch_season_data(year)
        batted = filter_batted_balls(df)

        # Keep only needed columns
        cols_available = [c for c in COLUMNS_TO_KEEP if c in batted.columns]
        batted = batted[cols_available]

        all_data.append(batted)

    # Combine all years
    combined = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal batted balls: {len(combined)}")

    # Save to parquet for efficiency
    output_path = os.path.join(DATA_DIR, 'statcast_batted_balls.parquet')
    combined.to_parquet(output_path, index=False)
    print(f"Saved to: {output_path}")

    # Also save a CSV for inspection
    csv_path = os.path.join(DATA_DIR, 'statcast_batted_balls.csv')
    combined.to_csv(csv_path, index=False)
    print(f"Also saved CSV: {csv_path}")

    return combined


if __name__ == '__main__':
    # Fetch 2021-2024 seasons for robust analysis
    years = [2021, 2022, 2023, 2024]
    df = process_and_save(years)

    # Print summary stats
    print("\n--- Summary ---")
    print(f"Date range: {df['game_date'].min()} to {df['game_date'].max()}")
    print(f"Unique games: {df['game_pk'].nunique()}")
    print(f"Unique stadiums (home teams): {df['home_team'].nunique()}")
    print(f"\nBatted balls per team (home games):")
    print(df['home_team'].value_counts())
