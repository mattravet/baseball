"""
Build distance prediction model and analyze residuals for marine layer effect.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(__file__).parent.parent / 'data'


def load_processed_data() -> pd.DataFrame:
    """Load the processed batted ball data."""
    df = pd.read_parquet(DATA_DIR / 'processed_batted_balls.parquet')
    df['game_date'] = pd.to_datetime(df['game_date'])
    df['month'] = df['game_date'].dt.month
    return df


def exploratory_analysis(df: pd.DataFrame):
    """Perform exploratory data analysis."""

    print("\n" + "="*70)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*70)

    # Key variable distributions
    print("\n--- Key Variable Statistics ---")
    key_vars = ['launch_speed', 'launch_angle', 'hit_distance_sc', 'humidity', 'temperature_f']
    print(df[key_vars].describe().round(2))

    # Correlation matrix
    print("\n--- Correlations with Hit Distance ---")
    correlations = df[key_vars + ['wind_speed', 'wind_direction']].corr()['hit_distance_sc'].sort_values(ascending=False)
    print(correlations.round(3))

    # Launch speed vs distance by batted ball type
    print("\n--- Distance by Batted Ball Type ---")
    print(df.groupby('bb_type')['hit_distance_sc'].agg(['count', 'mean', 'std']).round(1))

    # Seattle vs all comparison
    print("\n--- Seattle vs All Parks ---")
    sea = df[df['home_team'] == 'SEA']
    other = df[df['home_team'] != 'SEA']

    comparison = pd.DataFrame({
        'Seattle': [len(sea), sea['launch_speed'].mean(), sea['launch_angle'].mean(),
                    sea['hit_distance_sc'].mean(), sea['humidity'].mean()],
        'Other Parks': [len(other), other['launch_speed'].mean(), other['launch_angle'].mean(),
                        other['hit_distance_sc'].mean(), other['humidity'].mean()]
    }, index=['N', 'Avg Exit Velo', 'Avg Launch Angle', 'Avg Distance', 'Avg Humidity'])
    print(comparison.round(2))


def build_distance_model(df: pd.DataFrame) -> tuple:
    """
    Build a polynomial regression model for hit distance.

    Model: distance ~ exit_velocity + launch_angle + exit_velocity*launch_angle + launch_angle^2
    """

    print("\n" + "="*70)
    print("DISTANCE PREDICTION MODEL")
    print("="*70)

    # Prepare features
    X = df[['launch_speed', 'launch_angle']].copy()
    y = df['hit_distance_sc'].copy()

    # Add polynomial and interaction terms
    X['ev_la_interaction'] = X['launch_speed'] * X['launch_angle']
    X['la_squared'] = X['launch_angle'] ** 2

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    print("\n--- Model Performance ---")
    print(f"Training R²: {train_r2:.4f}")
    print(f"Test R²:     {test_r2:.4f}")
    print(f"Training RMSE: {train_rmse:.2f} feet")
    print(f"Test RMSE:     {test_rmse:.2f} feet")

    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"\n5-Fold CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

    # Coefficients
    print("\n--- Model Coefficients ---")
    coef_names = ['Exit Velocity', 'Launch Angle', 'EV*LA Interaction', 'LA²']
    for name, coef in zip(coef_names, model.coef_):
        print(f"  {name}: {coef:.4f}")
    print(f"  Intercept: {model.intercept_:.4f}")

    # Generate predictions for all data
    X_all = df[['launch_speed', 'launch_angle']].copy()
    X_all['ev_la_interaction'] = X_all['launch_speed'] * X_all['launch_angle']
    X_all['la_squared'] = X_all['launch_angle'] ** 2
    df['predicted_distance'] = model.predict(X_all)
    df['residual'] = df['hit_distance_sc'] - df['predicted_distance']

    return model, df


def analyze_residuals_by_stadium(df: pd.DataFrame):
    """Analyze residuals by stadium/home team."""

    print("\n" + "="*70)
    print("RESIDUAL ANALYSIS BY STADIUM")
    print("="*70)

    # Calculate mean residual by stadium
    stadium_residuals = df.groupby('home_team').agg({
        'residual': ['mean', 'std', 'count'],
        'humidity': 'mean',
        'temperature_f': 'mean'
    }).round(2)

    stadium_residuals.columns = ['mean_residual', 'std_residual', 'n_batted_balls',
                                  'avg_humidity', 'avg_temp']
    stadium_residuals = stadium_residuals.sort_values('mean_residual')

    print("\n--- Mean Residuals by Stadium (sorted) ---")
    print("Negative residual = ball travels SHORTER than predicted")
    print(stadium_residuals.to_string())

    # Highlight Seattle
    sea_rank = list(stadium_residuals.index).index('SEA') + 1
    sea_residual = stadium_residuals.loc['SEA', 'mean_residual']
    print(f"\n*** Seattle ranks #{sea_rank}/30 with mean residual of {sea_residual:.2f} feet ***")

    # Statistical test: Seattle vs all other parks
    sea_resid = df[df['home_team'] == 'SEA']['residual']
    other_resid = df[df['home_team'] != 'SEA']['residual']

    t_stat, p_value = stats.ttest_ind(sea_resid, other_resid)
    print(f"\nT-test (Seattle vs Others):")
    print(f"  Seattle mean residual: {sea_resid.mean():.2f} ft")
    print(f"  Other parks mean:      {other_resid.mean():.2f} ft")
    print(f"  Difference:            {sea_resid.mean() - other_resid.mean():.2f} ft")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value: {p_value:.4f}")

    if p_value < 0.05:
        print("  → Statistically significant difference (p < 0.05)")
    else:
        print("  → Not statistically significant (p >= 0.05)")

    return stadium_residuals


def analyze_residuals_by_month(df: pd.DataFrame):
    """Analyze Seattle residuals by month."""

    print("\n" + "="*70)
    print("SEATTLE RESIDUALS BY MONTH")
    print("="*70)

    sea = df[df['home_team'] == 'SEA']

    monthly = sea.groupby('month').agg({
        'residual': ['mean', 'std', 'count'],
        'humidity': 'mean',
        'temperature_f': 'mean',
        'marine_layer_likely': 'mean'
    }).round(2)

    monthly.columns = ['mean_residual', 'std_residual', 'n', 'avg_humidity',
                        'avg_temp', 'pct_marine_layer']
    monthly['pct_marine_layer'] = (monthly['pct_marine_layer'] * 100).round(1)

    print("\nMarine layer is expected to be strongest in May-July")
    print(monthly.to_string())

    # Correlation between residual and marine layer conditions
    corr_humidity = sea['residual'].corr(sea['humidity'])
    print(f"\nCorrelation (Seattle residual vs humidity): {corr_humidity:.3f}")


def analyze_residuals_by_weather(df: pd.DataFrame):
    """Analyze residuals by weather conditions."""

    print("\n" + "="*70)
    print("RESIDUAL ANALYSIS BY WEATHER CONDITIONS")
    print("="*70)

    # Seattle data
    sea = df[df['home_team'] == 'SEA']

    # High vs low humidity
    high_humid = sea[sea['humidity'] >= 70]['residual']
    low_humid = sea[sea['humidity'] < 70]['residual']

    print("\n--- Seattle: High vs Low Humidity ---")
    print(f"High humidity (≥70%): mean residual = {high_humid.mean():.2f} ft (n={len(high_humid)})")
    print(f"Low humidity (<70%):  mean residual = {low_humid.mean():.2f} ft (n={len(low_humid)})")

    if len(high_humid) > 10 and len(low_humid) > 10:
        t_stat, p_value = stats.ttest_ind(high_humid, low_humid)
        print(f"t-test p-value: {p_value:.4f}")

    # Marine layer condition
    ml_yes = sea[sea['marine_layer_likely'] == True]['residual']
    ml_no = sea[sea['marine_layer_likely'] == False]['residual']

    print("\n--- Seattle: Marine Layer Likely vs Not ---")
    print(f"Marine layer likely:     mean residual = {ml_yes.mean():.2f} ft (n={len(ml_yes)})")
    print(f"Marine layer not likely: mean residual = {ml_no.mean():.2f} ft (n={len(ml_no)})")

    if len(ml_yes) > 10 and len(ml_no) > 10:
        t_stat, p_value = stats.ttest_ind(ml_yes, ml_no)
        print(f"t-test p-value: {p_value:.4f}")

    # Wind direction analysis
    print("\n--- Seattle: Residuals by Wind Direction ---")
    # Categorize wind direction
    def wind_category(deg):
        if pd.isna(deg):
            return 'Unknown'
        elif 315 <= deg or deg < 45:
            return 'N (Onshore)'
        elif 45 <= deg < 135:
            return 'E'
        elif 135 <= deg < 225:
            return 'S'
        else:
            return 'W (Onshore)'

    sea_wind = sea.copy()
    sea_wind['wind_cat'] = sea_wind['wind_direction'].apply(wind_category)
    print(sea_wind.groupby('wind_cat')['residual'].agg(['mean', 'count']).round(2))


def analyze_residuals_by_spray_angle(df: pd.DataFrame):
    """Analyze residuals by spray angle (field direction)."""

    print("\n" + "="*70)
    print("RESIDUAL ANALYSIS BY SPRAY ANGLE")
    print("="*70)

    sea = df[df['home_team'] == 'SEA']

    # By spray zone
    print("\n--- Seattle Residuals by Spray Zone ---")
    spray_analysis = sea.groupby('spray_zone')['residual'].agg(['mean', 'std', 'count']).round(2)
    print(spray_analysis)

    # Compare to all parks
    print("\n--- All Parks Residuals by Spray Zone ---")
    all_spray = df.groupby('spray_zone')['residual'].agg(['mean', 'std', 'count']).round(2)
    print(all_spray)


def run_regression_with_park_effects(df: pd.DataFrame):
    """Run regression with park fixed effects to isolate Seattle effect."""

    print("\n" + "="*70)
    print("REGRESSION MODEL WITH PARK EFFECTS")
    print("="*70)

    # Create dummy variables for parks (using Seattle as reference)
    df_model = df.copy()

    # Features: exit velo, launch angle, interaction, humidity, temp, Seattle indicator
    df_model['is_seattle'] = (df_model['home_team'] == 'SEA').astype(int)
    df_model['ev_la'] = df_model['launch_speed'] * df_model['launch_angle']
    df_model['la_sq'] = df_model['launch_angle'] ** 2

    # Model with weather and Seattle indicator
    X = df_model[['launch_speed', 'launch_angle', 'ev_la', 'la_sq',
                   'humidity', 'temperature_f', 'is_seattle']]
    X = sm.add_constant(X)
    y = df_model['hit_distance_sc']

    model = sm.OLS(y, X).fit()

    print("\n--- OLS Regression Results (abbreviated) ---")
    print(f"R-squared: {model.rsquared:.4f}")
    print(f"Observations: {model.nobs:.0f}")
    print("\nKey Coefficients:")
    print(f"  Seattle effect: {model.params['is_seattle']:.2f} ft (p={model.pvalues['is_seattle']:.4f})")
    print(f"  Humidity:       {model.params['humidity']:.3f} ft per 1% (p={model.pvalues['humidity']:.4f})")
    print(f"  Temperature:    {model.params['temperature_f']:.3f} ft per 1°F (p={model.pvalues['temperature_f']:.4f})")

    # Interpretation
    print("\n--- Interpretation ---")
    seattle_coef = model.params['is_seattle']
    seattle_pval = model.pvalues['is_seattle']

    if seattle_pval < 0.05:
        if seattle_coef < 0:
            print(f"After controlling for exit velocity, launch angle, humidity, and temperature,")
            print(f"balls hit in Seattle travel {abs(seattle_coef):.1f} feet SHORTER than predicted.")
            print(f"This is statistically significant (p={seattle_pval:.4f}).")
        else:
            print(f"Seattle shows a POSITIVE effect of {seattle_coef:.1f} feet.")
    else:
        print(f"Seattle effect ({seattle_coef:.1f} ft) is not statistically significant (p={seattle_pval:.2f}).")

    return model


def main():
    """Main analysis pipeline."""

    # Load data
    print("Loading processed data...")
    df = load_processed_data()
    print(f"Loaded {len(df)} batted balls")

    # EDA
    exploratory_analysis(df)

    # Build model and calculate residuals
    model, df = build_distance_model(df)

    # Analyze residuals
    analyze_residuals_by_stadium(df)
    analyze_residuals_by_month(df)
    analyze_residuals_by_weather(df)
    analyze_residuals_by_spray_angle(df)

    # Regression with park effects
    run_regression_with_park_effects(df)

    # Save data with residuals
    output_path = DATA_DIR / 'analysis_results.parquet'
    df.to_parquet(output_path, index=False)
    print(f"\nSaved results to: {output_path}")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

    return df


if __name__ == '__main__':
    df = main()
