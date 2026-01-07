"""
Create visualizations for the marine layer analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / 'data'
OUTPUT_DIR = Path(__file__).parent.parent / 'figures'
OUTPUT_DIR.mkdir(exist_ok=True)


def load_data():
    """Load analysis results."""
    df = pd.read_parquet(DATA_DIR / 'analysis_results.parquet')
    df['game_date'] = pd.to_datetime(df['game_date'])
    df['month'] = df['game_date'].dt.month
    return df


def plot_stadium_residuals(df: pd.DataFrame):
    """Plot mean residuals by stadium."""

    fig, ax = plt.subplots(figsize=(12, 8))

    # Calculate mean residuals
    stadium_resid = df.groupby('home_team')['residual'].mean().sort_values()

    # Color Seattle differently
    colors = ['#2ecc71' if x == 'SEA' else
              '#e74c3c' if x == 'COL' else  # Colorado (altitude)
              '#3498db' if x == 'SF' else   # San Francisco (marine layer)
              '#95a5a6' for x in stadium_resid.index]

    bars = ax.barh(stadium_resid.index, stadium_resid.values, color=colors)

    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Mean Residual (feet)', fontsize=12)
    ax.set_ylabel('Stadium (Home Team)', fontsize=12)
    ax.set_title('Hit Distance Residuals by Stadium\n(Negative = Ball Travels Shorter Than Predicted)', fontsize=14)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='Seattle'),
        Patch(facecolor='#3498db', label='San Francisco (marine layer)'),
        Patch(facecolor='#e74c3c', label='Colorado (altitude)'),
        Patch(facecolor='#95a5a6', label='Other stadiums')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'stadium_residuals.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'stadium_residuals.png'}")


def plot_seattle_monthly(df: pd.DataFrame):
    """Plot Seattle residuals by month."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sea = df[df['home_team'] == 'SEA']

    # Left: Residuals by month
    monthly = sea.groupby('month').agg({
        'residual': ['mean', 'std', 'count']
    })
    monthly.columns = ['mean', 'std', 'count']
    monthly['se'] = monthly['std'] / np.sqrt(monthly['count'])

    ax1 = axes[0]
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']  # Apr, May, Jun, Jul
    bars = ax1.bar(monthly.index, monthly['mean'], yerr=monthly['se']*1.96,
                   capsize=5, color=colors, edgecolor='black')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_xlabel('Month', fontsize=12)
    ax1.set_ylabel('Mean Residual (feet)', fontsize=12)
    ax1.set_title('Seattle: Mean Residual by Month\n(with 95% CI)', fontsize=14)
    ax1.set_xticks([4, 5, 6, 7])
    ax1.set_xticklabels(['April', 'May', 'June', 'July'])

    # Right: Weather conditions by month
    ax2 = axes[1]
    monthly_weather = sea.groupby('month').agg({
        'humidity': 'mean',
        'temperature_f': 'mean',
        'marine_layer_likely': 'mean'
    })

    x = np.arange(4)
    width = 0.25

    ax2.bar(x - width, monthly_weather['humidity'], width, label='Humidity (%)', color='#3498db')
    ax2.bar(x, monthly_weather['temperature_f'], width, label='Temp (°F)', color='#e74c3c')
    ax2.bar(x + width, monthly_weather['marine_layer_likely']*100, width,
            label='Marine Layer %', color='#2ecc71')

    ax2.set_xlabel('Month', fontsize=12)
    ax2.set_ylabel('Value', fontsize=12)
    ax2.set_title('Seattle: Weather Conditions by Month', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(['April', 'May', 'June', 'July'])
    ax2.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'seattle_monthly.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'seattle_monthly.png'}")


def plot_humidity_effect(df: pd.DataFrame):
    """Plot relationship between humidity and residuals."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: All parks - humidity vs residual
    ax1 = axes[0]
    # Bin humidity and calculate mean residual
    df['humidity_bin'] = pd.cut(df['humidity'], bins=np.arange(0, 110, 10))
    humidity_effect = df.groupby('humidity_bin')['residual'].agg(['mean', 'count'])

    # Filter bins with sufficient data
    humidity_effect = humidity_effect[humidity_effect['count'] >= 100]

    bin_centers = [5 + i*10 for i in range(len(humidity_effect))]
    ax1.bar(bin_centers[:len(humidity_effect)], humidity_effect['mean'], width=8,
            color='#3498db', edgecolor='black', alpha=0.7)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_xlabel('Humidity (%)', fontsize=12)
    ax1.set_ylabel('Mean Residual (feet)', fontsize=12)
    ax1.set_title('All Parks: Hit Distance Residual by Humidity', fontsize=14)

    # Right: Seattle specifically
    ax2 = axes[1]
    sea = df[df['home_team'] == 'SEA']
    ax2.scatter(sea['humidity'], sea['residual'], alpha=0.3, s=20, color='#2ecc71')

    # Add regression line
    z = np.polyfit(sea['humidity'], sea['residual'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(sea['humidity'].min(), sea['humidity'].max(), 100)
    ax2.plot(x_line, p(x_line), 'r-', linewidth=2, label=f'Trend: {z[0]:.2f}x + {z[1]:.1f}')

    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Humidity (%)', fontsize=12)
    ax2.set_ylabel('Residual (feet)', fontsize=12)
    ax2.set_title('Seattle: Residual vs Humidity', fontsize=14)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'humidity_effect.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'humidity_effect.png'}")


def plot_wind_direction_effect(df: pd.DataFrame):
    """Plot residuals by wind direction for Seattle."""

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

    sea = df[df['home_team'] == 'SEA'].copy()

    # Bin wind direction (every 45 degrees)
    sea['wind_bin'] = pd.cut(sea['wind_direction'],
                              bins=[0, 45, 90, 135, 180, 225, 270, 315, 360],
                              labels=['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])

    wind_resid = sea.groupby('wind_bin')['residual'].agg(['mean', 'count'])

    # Convert to radians
    angles = np.array([0, 45, 90, 135, 180, 225, 270, 315]) * np.pi / 180
    angles = np.append(angles, angles[0])  # Close the circle

    values = list(wind_resid['mean'])
    values.append(values[0])  # Close the circle

    # Color based on positive/negative
    colors = ['#e74c3c' if v < 0 else '#2ecc71' for v in values[:-1]]

    ax.bar(angles[:-1], np.abs(values[:-1]), width=0.7, alpha=0.7,
           color=colors, edgecolor='black')

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315],
                       ['N\n(Onshore)', 'NE', 'E', 'SE', 'S', 'SW', 'W\n(Onshore)', 'NW'])
    ax.set_title('Seattle: Mean Residual by Wind Direction\n(Red=Negative, Green=Positive)', fontsize=14, pad=20)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'wind_direction.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'wind_direction.png'}")


def plot_model_performance(df: pd.DataFrame):
    """Plot model predictions vs actual."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Predicted vs Actual
    ax1 = axes[0]
    sample = df.sample(n=min(5000, len(df)), random_state=42)
    ax1.scatter(sample['predicted_distance'], sample['hit_distance_sc'],
                alpha=0.2, s=10, c='#3498db')
    ax1.plot([0, 500], [0, 500], 'r--', linewidth=2, label='Perfect prediction')
    ax1.set_xlabel('Predicted Distance (feet)', fontsize=12)
    ax1.set_ylabel('Actual Distance (feet)', fontsize=12)
    ax1.set_title('Model: Predicted vs Actual Distance\n(R² = 0.85)', fontsize=14)
    ax1.legend()
    ax1.set_xlim(0, 500)
    ax1.set_ylim(0, 500)

    # Right: Residual distribution
    ax2 = axes[1]
    ax2.hist(df['residual'], bins=50, color='#3498db', edgecolor='black', alpha=0.7)
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax2.axvline(x=df['residual'].mean(), color='green', linestyle='-', linewidth=2,
                label=f'Mean: {df["residual"].mean():.1f} ft')
    ax2.set_xlabel('Residual (feet)', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Distribution of Residuals\n(Actual - Predicted Distance)', fontsize=14)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'model_performance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'model_performance.png'}")


def plot_spray_angle_effect(df: pd.DataFrame):
    """Plot residuals by spray angle for Seattle vs all."""

    fig, ax = plt.subplots(figsize=(10, 6))

    sea = df[df['home_team'] == 'SEA']
    other = df[df['home_team'] != 'SEA']

    # Calculate by spray zone
    sea_spray = sea.groupby('spray_zone')['residual'].mean()
    other_spray = other.groupby('spray_zone')['residual'].mean()

    x = np.arange(3)
    width = 0.35

    ax.bar(x - width/2, sea_spray, width, label='Seattle', color='#2ecc71', edgecolor='black')
    ax.bar(x + width/2, other_spray, width, label='Other Parks', color='#95a5a6', edgecolor='black')

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Spray Direction', fontsize=12)
    ax.set_ylabel('Mean Residual (feet)', fontsize=12)
    ax.set_title('Mean Residual by Spray Direction\nSeattle vs Other Parks', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(['Pull Side', 'Center Field', 'Opposite Field'])
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'spray_angle.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'spray_angle.png'}")


def create_summary_figure(df: pd.DataFrame):
    """Create a summary figure with key findings."""

    fig = plt.figure(figsize=(16, 12))

    # Stadium comparison (top left)
    ax1 = fig.add_subplot(2, 2, 1)
    stadium_resid = df.groupby('home_team')['residual'].mean().sort_values()
    colors = ['#2ecc71' if x == 'SEA' else
              '#e74c3c' if x == 'COL' else
              '#3498db' if x == 'SF' else
              '#95a5a6' for x in stadium_resid.index]
    ax1.barh(stadium_resid.index, stadium_resid.values, color=colors)
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_xlabel('Mean Residual (feet)')
    ax1.set_title('Residuals by Stadium')

    # Seattle monthly (top right)
    ax2 = fig.add_subplot(2, 2, 2)
    sea = df[df['home_team'] == 'SEA']
    monthly = sea.groupby('month')['residual'].mean()
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
    ax2.bar(monthly.index, monthly.values, color=colors, edgecolor='black')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Mean Residual (feet)')
    ax2.set_title('Seattle: Residuals by Month')
    ax2.set_xticks([4, 5, 6, 7])
    ax2.set_xticklabels(['Apr', 'May', 'Jun', 'Jul'])

    # Model performance (bottom left)
    ax3 = fig.add_subplot(2, 2, 3)
    sample = df.sample(n=min(3000, len(df)), random_state=42)
    ax3.scatter(sample['predicted_distance'], sample['hit_distance_sc'], alpha=0.2, s=5)
    ax3.plot([0, 500], [0, 500], 'r--', linewidth=2)
    ax3.set_xlabel('Predicted Distance (feet)')
    ax3.set_ylabel('Actual Distance (feet)')
    ax3.set_title('Model Performance (R² = 0.85)')

    # Text summary (bottom right)
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')

    summary_text = """
    MARINE LAYER ANALYSIS SUMMARY

    Model: Distance ~ Exit Velocity + Launch Angle
    R² = 0.85, RMSE = 50 feet

    KEY FINDINGS:

    1. Seattle Overall: Rank #16/30
       Mean residual: -0.29 ft (not significant)

    2. Monthly Pattern:
       - May shows strongest suppression: -5.69 ft
       - July shows positive effect: +4.82 ft

    3. Weather Effects:
       - High humidity: -1.92 ft vs +0.25 ft low
       - N/Onshore winds: -2.63 ft
       - S winds: +2.82 ft

    4. Humidity Coefficient: -0.038 ft per 1%
       (Statistically significant, p=0.03)

    5. Comparison:
       - Colorado (altitude): +8.88 ft
       - San Francisco (marine): -3.79 ft

    CONCLUSION: Directional support for marine layer
    effect, but not statistically significant with
    this sample size (n=770 Seattle batted balls).
    """

    ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Seattle Marine Layer Effect on Batted Ball Distance', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'summary.png'}")


def main():
    """Generate all visualizations."""

    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df)} records")

    print("\nGenerating visualizations...")
    plot_model_performance(df)
    plot_stadium_residuals(df)
    plot_seattle_monthly(df)
    plot_humidity_effect(df)
    plot_wind_direction_effect(df)
    plot_spray_angle_effect(df)
    create_summary_figure(df)

    print(f"\nAll figures saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
