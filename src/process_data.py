"""
Process and merge Statcast batted ball data with weather data.
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / 'data'

# Map MLB team abbreviations to city names in weather data
# Weather data cities: Vancouver, Portland, San Francisco, Seattle, Los Angeles, San Diego,
# Las Vegas, Phoenix, Albuquerque, Denver, San Antonio, Dallas, Houston, Kansas City,
# Minneapolis, Saint Louis, Chicago, Nashville, Indianapolis, Atlanta, Detroit, Jacksonville,
# Charlotte, Miami, Pittsburgh, Toronto, Philadelphia, New York, Montreal, Boston
TEAM_TO_WEATHER_CITY = {
    'SEA': 'Seattle',
    'SF': 'San Francisco',
    'OAK': 'San Francisco',  # Oakland - use SF weather (nearby)
    'LAD': 'Los Angeles',
    'LAA': 'Los Angeles',
    'SD': 'San Diego',
    'ARI': 'Phoenix',
    'COL': 'Denver',
    'TEX': 'Dallas',
    'HOU': 'Houston',
    'KC': 'Kansas City',
    'MIN': 'Minneapolis',
    'STL': 'Saint Louis',
    'CHC': 'Chicago',
    'CWS': 'Chicago',
    'MIL': 'Chicago',  # Milwaukee - use Chicago (nearby)
    'CIN': 'Indianapolis',  # Cincinnati - use Indianapolis (nearby)
    'ATL': 'Atlanta',
    'DET': 'Detroit',
    'CLE': 'Detroit',  # Cleveland - use Detroit (nearby)
    'MIA': 'Miami',
    'TB': 'Miami',  # Tampa Bay - use Miami (Florida)
    'PIT': 'Pittsburgh',
    'TOR': 'Toronto',
    'PHI': 'Philadelphia',
    'NYY': 'New York',
    'NYM': 'New York',
    'BOS': 'Boston',
    'BAL': 'Philadelphia',  # Baltimore - use Philadelphia (nearby)
    'WSH': 'Philadelphia',  # Washington DC - use Philadelphia (nearby)
}


def load_weather_data() -> pd.DataFrame:
    """Load and combine all weather CSV files into a single DataFrame."""

    # Load each weather variable
    humidity = pd.read_csv(DATA_DIR / 'humidity.csv')
    temperature = pd.read_csv(DATA_DIR / 'temperature.csv')
    wind_direction = pd.read_csv(DATA_DIR / 'wind_direction.csv')
    wind_speed = pd.read_csv(DATA_DIR / 'wind_speed.csv')
    pressure = pd.read_csv(DATA_DIR / 'pressure.csv')

    # Melt each to long format and merge
    def melt_weather(df, value_name):
        df['datetime'] = pd.to_datetime(df['datetime'])
        melted = df.melt(id_vars=['datetime'], var_name='city', value_name=value_name)
        return melted

    humidity_long = melt_weather(humidity, 'humidity')
    temp_long = melt_weather(temperature, 'temperature_k')
    wind_dir_long = melt_weather(wind_direction, 'wind_direction')
    wind_spd_long = melt_weather(wind_speed, 'wind_speed')
    pressure_long = melt_weather(pressure, 'pressure')

    # Merge all weather data
    weather = humidity_long.merge(temp_long, on=['datetime', 'city'])
    weather = weather.merge(wind_dir_long, on=['datetime', 'city'])
    weather = weather.merge(wind_spd_long, on=['datetime', 'city'])
    weather = weather.merge(pressure_long, on=['datetime', 'city'])

    # Convert temperature from Kelvin to Fahrenheit
    weather['temperature_f'] = (weather['temperature_k'] - 273.15) * 9/5 + 32

    # Extract date and hour for merging
    weather['date'] = weather['datetime'].dt.date
    weather['hour'] = weather['datetime'].dt.hour

    return weather


def calculate_spray_angle(hc_x: pd.Series, hc_y: pd.Series) -> pd.Series:
    """
    Calculate spray angle from Statcast hit coordinates.

    Statcast coordinates:
    - hc_x: horizontal position (0 = left field line, 125.42 = center, 250 = right field line)
    - hc_y: vertical position (0 = home plate, increases toward outfield)

    Spray angle:
    - Negative = pull side for RHB (left field), positive = opposite field
    - Range approximately -45 to +45 degrees
    """
    # Center is at x=125.42 (approximately)
    CENTER_X = 125.42

    # Calculate angle from center (in degrees)
    # Positive x values (> center) = right field
    # Negative x values (< center) = left field
    dx = hc_x - CENTER_X
    dy = hc_y  # distance from home plate

    # Avoid division by zero
    spray_angle = np.degrees(np.arctan2(dx, dy))

    return spray_angle


def get_game_hour_estimate(inning: int, inning_topbot: str) -> int:
    """
    Estimate the hour of day for a pitch based on inning.

    Assumes:
    - Night games start around 7 PM (19:00) local time
    - Day games start around 1 PM (13:00) local time
    - Each half-inning takes ~15-20 minutes

    For simplicity, we'll assume evening games (most common) starting at 7 PM.
    """
    # Base start time (7 PM = 19:00)
    base_hour = 19

    # Calculate elapsed half-innings
    half_innings = (inning - 1) * 2
    if inning_topbot == 'Bot':
        half_innings += 1

    # Estimate ~18 minutes per half-inning = ~0.3 hours
    elapsed_hours = half_innings * 0.3

    estimated_hour = int(base_hour + elapsed_hours) % 24
    return estimated_hour


def create_marine_layer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features that indicate marine layer conditions.

    Marine layer characteristics:
    - High humidity (>70%)
    - Moderate temperatures (50-70°F)
    - Wind from W/NW/N (onshore wind in Seattle: ~270-360 degrees)
    """
    df = df.copy()

    # High humidity indicator
    df['high_humidity'] = df['humidity'] >= 70

    # Moderate temperature (typical marine layer temps)
    df['marine_temp_range'] = (df['temperature_f'] >= 50) & (df['temperature_f'] <= 70)

    # Onshore wind (W to N, roughly 270-360 degrees)
    # Also include NW winds around 315 degrees
    df['onshore_wind'] = (
        (df['wind_direction'] >= 270) |
        (df['wind_direction'] <= 45)  # Wrap around for N wind
    )

    # Combined marine layer indicator
    df['marine_layer_likely'] = (
        df['high_humidity'] &
        df['marine_temp_range']
    )

    return df


def process_and_merge_data():
    """Main function to process and merge all data."""

    print("Loading Statcast data...")
    statcast = pd.read_csv(DATA_DIR / 'savant_data.csv')
    statcast['game_date'] = pd.to_datetime(statcast['game_date'])
    print(f"  Loaded {len(statcast)} batted balls")

    print("\nLoading weather data...")
    weather = load_weather_data()
    print(f"  Loaded {len(weather)} hourly weather records")

    print("\nCalculating spray angles...")
    statcast['spray_angle'] = calculate_spray_angle(statcast['hc_x'], statcast['hc_y'])

    # Categorize spray angle into LF/CF/RF
    statcast['spray_zone'] = pd.cut(
        statcast['spray_angle'],
        bins=[-90, -15, 15, 90],
        labels=['pull', 'center', 'oppo']
    )

    print("\nEstimating game hours and merging weather...")
    # Map home team to weather city
    statcast['weather_city'] = statcast['home_team'].map(TEAM_TO_WEATHER_CITY)

    # Estimate hour of pitch
    statcast['estimated_hour'] = statcast.apply(
        lambda row: get_game_hour_estimate(row['inning'], row['inning_topbot']),
        axis=1
    )

    # Prepare for merge
    statcast['date'] = statcast['game_date'].dt.date

    # Merge weather data
    merged = statcast.merge(
        weather,
        left_on=['date', 'weather_city', 'estimated_hour'],
        right_on=['date', 'city', 'hour'],
        how='left'
    )

    # Create marine layer features
    print("\nCreating marine layer features...")
    merged = create_marine_layer_features(merged)

    # Filter to valid data for modeling
    print("\nFiltering to complete cases...")
    required_cols = ['launch_speed', 'launch_angle', 'hit_distance_sc',
                     'humidity', 'temperature_f', 'wind_direction']
    merged_valid = merged.dropna(subset=required_cols)

    print(f"  Valid records: {len(merged_valid)} / {len(merged)}")

    # Save processed data
    output_path = DATA_DIR / 'processed_batted_balls.parquet'
    merged_valid.to_parquet(output_path, index=False)
    print(f"\nSaved to: {output_path}")

    # Also save CSV for inspection
    csv_path = DATA_DIR / 'processed_batted_balls.csv'
    merged_valid.to_csv(csv_path, index=False)
    print(f"Also saved: {csv_path}")

    return merged_valid


def print_summary(df: pd.DataFrame):
    """Print summary statistics of the processed data."""

    print("\n" + "="*60)
    print("PROCESSED DATA SUMMARY")
    print("="*60)

    print(f"\nTotal batted balls: {len(df)}")
    print(f"Date range: {df['game_date'].min()} to {df['game_date'].max()}")
    print(f"Unique games: {df['game_pk'].nunique()}")
    print(f"Stadiums (home teams): {df['home_team'].nunique()}")

    print("\n--- Seattle Data ---")
    sea = df[df['home_team'] == 'SEA']
    print(f"Seattle batted balls: {len(sea)}")
    print(f"Seattle games: {sea['game_pk'].nunique()}")
    print(f"Seattle months: {sea['game_date'].dt.month.value_counts().sort_index().to_dict()}")

    print("\n--- Weather Coverage ---")
    print(f"Records with weather: {df['humidity'].notna().sum()}")
    print(f"Humidity range: {df['humidity'].min():.1f} - {df['humidity'].max():.1f}%")
    print(f"Temperature range: {df['temperature_f'].min():.1f} - {df['temperature_f'].max():.1f}°F")

    print("\n--- Marine Layer Indicators (Seattle) ---")
    if len(sea) > 0:
        print(f"High humidity games: {sea['high_humidity'].sum()} ({100*sea['high_humidity'].mean():.1f}%)")
        print(f"Onshore wind: {sea['onshore_wind'].sum()} ({100*sea['onshore_wind'].mean():.1f}%)")
        print(f"Marine layer likely: {sea['marine_layer_likely'].sum()} ({100*sea['marine_layer_likely'].mean():.1f}%)")

    print("\n--- Spray Angle Distribution ---")
    print(df['spray_zone'].value_counts())


if __name__ == '__main__':
    df = process_and_merge_data()
    print_summary(df)
