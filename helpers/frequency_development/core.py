"""
Core functionality for frequency development analysis
"""

import os
import json
import pandas as pd
from .constants import *
import numpy as np
from typing import Dict
import shutil
import numpy as np
import os
from pathlib import Path
from dateutil.relativedelta import relativedelta

from .plot_utils import plot_development_factors, plot_forecast_trend


def calculate_month_difference(start_date_column, end_date_column, row):
    """Compute the month difference between two dates in a row."""
    start_date = row[start_date_column]
    end_date = row[end_date_column]
    if pd.isna(start_date) or pd.isna(end_date):
        return None
    return relativedelta(end_date, start_date).years * 12 + relativedelta(end_date, start_date).months


# --- Config utility functions ---
def ensure_user_config(config_path, default_config_path=None):
    """
    Ensure a user-writable config file exists at config_path.
    If not, copy from default_config_path or create an empty config.
    """
    if not os.path.exists(config_path):
        if default_config_path and os.path.exists(default_config_path):
            shutil.copy(default_config_path, config_path)
        else:
            with open(config_path, "w") as f:
                json.dump({}, f, indent=2)

def load_user_config(config_path, default_config_path=None):
    """Load user config from a JSON file, ensuring it exists and is user-writable."""
    ensure_user_config(config_path, default_config_path)
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    else:
        return {}

def get_or_create_config_for_key(all_configs, segment, cutoff_date):
    """
    Given all_configs (dict), segment (str), and cutoff_date (str or pd.Timestamp),
    returns a config dict for the key segment__cutoff_date. If not found, tries to find the most recent prior config for the segment.
    If still not found, returns a default config dict.
    """
    config_key = f"{segment}__{str(cutoff_date)}"
    user_config = all_configs.get(config_key, None)
    if user_config is None:
        # Find all keys for this segment
        keys = [k for k in all_configs if k.startswith(f"{segment}__")]
        # Extract dates and filter those before the current cutoff_date
        prior_dates = []
        for k in keys:
            try:
                seg, date_str = k.split("__")
                date_obj = pd.to_datetime(date_str)
                if date_obj < pd.to_datetime(cutoff_date):
                    prior_dates.append((date_obj, k))
            except Exception:
                continue
        if prior_dates:
            prior_dates.sort(reverse=True)
            user_config = all_configs[prior_dates[0][1]].copy()
        else:
            user_config = {"num_claims_threshold": 100,"exclude_dates": [], "min_max_development": 0, "manual_overrides": []}
        all_configs[config_key] = user_config
    else:
        if 'manual_overrides' not in user_config:
            user_config['manual_overrides'] = []
    return user_config

def save_user_config(config_path, all_configs):
    """Save user config to a JSON file."""
    with open(config_path, "w") as f:
        json.dump(all_configs, f, indent=2)

def get_manual_overrides(all_configs, config_key):
    """Get manual overrides for a segment/cutoff from config."""
    return all_configs.get(config_key, {}).get('manual_overrides', [])

def set_manual_overrides(all_configs, config_key, manual_overrides):
    """Set manual overrides for a segment/cutoff in config."""
    if config_key not in all_configs:
        all_configs[config_key] = {}
    all_configs[config_key]['manual_overrides'] = manual_overrides

# --- Data loading ---
def load_data():
    """Load and preprocess data (policies and claims)."""
    return load_and_preprocess_data()

def _load_local_data_pandas():
    """
    Load local data from csv files.
    """
    policies = pd.read_csv('_data/policies.csv', parse_dates=[SOLD_DAY, DEPARTURE_DAY, DATE_SOLD_END_OF_MONTH, DATE_DEPART_END_OF_MONTH])
    claims = pd.read_csv('_data/claims.csv', parse_dates=[SOLD_DAY, DEPARTURE_DAY, RECEIVED_DAY, DATE_SOLD_END_OF_MONTH, DATE_DEPART_END_OF_MONTH, DATE_RECEIVED_END_OF_MONTH])
    return policies, claims



def compute_claim_development_triangle_pandas(df_claim: pd.DataFrame,groupby_cols: list[str] = [SEGMENT, DATE_DEPART_END_OF_MONTH, DEVELOPMENT]) -> pd.DataFrame:
    """
    Computes number of claims by cohort (based on departure month) and development month
    (received - departure), allowing for negative dev months (e.g., cancellation before trip).
    
    Parameters:
    - df_claim: A pandas DataFrame with 'idpol', 'dateDepart', 'dateReceived'.
    
    Returns:
    - DataFrame with: cohort_month, development_month, claim_count
    """
    # Calculate development month
    df_claim[DEVELOPMENT] = ((df_claim[RECEIVED_DAY] - df_claim[DEPARTURE_DAY])
                                   .dt.days / 30.44).round().astype(int)

    # Aggregate to triangle structure
    df_dev_triangle = (df_claim
        .groupby(groupby_cols)
        .agg({CLAIM_COUNT: 'sum'})
        .reset_index()
        .sort_values(groupby_cols)
    )

    return df_dev_triangle

def compute_frequency_dev_pandas(
    df_claim_dev: pd.DataFrame,
    df_pol_count: pd.DataFrame,
    groupby_cols: list[str] = [SEGMENT, DATE_DEPART_END_OF_MONTH]
) -> pd.DataFrame:
    """
    Merge claims and policy counts, compute frequency and cumulative frequency.
    """
    # Join on TargetFileDescr and cohort_month
    df = pd.merge(
        df_claim_dev,
        df_pol_count,
        on=[SEGMENT, DATE_DEPART_END_OF_MONTH],
        how='left'
    )
    
    # Compute frequency per dev month
    df[FREQUENCY_VAL] = df[CLAIM_COUNT] / df[POLICY_COUNT]
    
    # Compute cumulative values
    df = df.sort_values(groupby_cols + [DEVELOPMENT])
    
    df[CLAIM_CUMUL] = df.groupby(groupby_cols)[CLAIM_COUNT].cumsum()
    df[FREQUENCY_CUMUL] = df[CLAIM_CUMUL] / df[POLICY_COUNT] 
    return df


def _light_feature_engineering_pandas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add light feature engineering to the dataframe.
    """
    df = df.copy()
    df[DEPARTURE_YEAR] = df[DATE_DEPART_END_OF_MONTH].dt.year
    df[DEPARTURE_MONTH] = df[DATE_DEPART_END_OF_MONTH].dt.month
    
    # Normalize dates to first day of month
    df[DATE_SOLD_END_OF_MONTH] = df[SOLD_DAY].dt.to_period('M').dt.to_timestamp()
    df[DATE_DEPART_END_OF_MONTH] = df[DEPARTURE_DAY].dt.to_period('M').dt.to_timestamp()
    if RECEIVED_DAY in df.columns:
        df[DATE_RECEIVED_END_OF_MONTH] = df[RECEIVED_DAY].dt.to_period('M').dt.to_timestamp()
    
    return df

def _light_feature_engineering_pandas_tripmate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add light feature engineering to the dataframe.
    """
    df = df.copy()
    df[DEPARTURE_YEAR] = df[DATE_DEPART_END_OF_MONTH].dt.year
    df[DEPARTURE_MONTH] = df[DATE_DEPART_END_OF_MONTH].dt.month
    
    # Normalize dates to first day of month
    #df[DATE_SOLD_END_OF_MONTH] = df[SOLD_DAY].dt.to_period('M').dt.to_timestamp()
    df[DATE_DEPART_END_OF_MONTH] = df[DEPARTURE_DAY].dt.to_period('M').dt.to_timestamp()
    if RECEIVED_DAY in df.columns:
        df[DATE_RECEIVED_END_OF_MONTH] = df[RECEIVED_DAY].dt.to_period('M').dt.to_timestamp()
    
    return df

def load_and_preprocess_data():
    """
    Load local data from csv files and apply light feature engineering.
    """
    policies, claims = _load_local_data_pandas()
    policies = _light_feature_engineering_pandas(policies)
    claims = _light_feature_engineering_pandas(claims)

    # Handle various corner cases for CAT_CODE:
    # 1. Replace None/NA values with "baseline"
    # 2. Replace empty strings with "baseline" 
    # 3. Strip whitespace from values
    # 4. Convert to lowercase for consistency
    claims[CAT_CODE] = claims[CAT_CODE].fillna("baseline")
    claims[CAT_CODE] = claims[CAT_CODE].replace("0", "baseline")
    claims[CAT_CODE] = claims[CAT_CODE].replace("", "baseline")
    claims[CAT_CODE] = claims[CAT_CODE].str.strip()
    claims[CAT_CODE] = claims[CAT_CODE].str.lower()
    

    #rename legacy columns names
    policies = policies.rename(columns={POLICY_SOLD_COUNT: POLICY_COUNT, 'TargetFileDescr': SEGMENT})
    claims = claims.rename(columns={CLAIM_NUM_UNIQUE: CLAIM_COUNT, 'TargetFileDescr': SEGMENT})

    return policies, claims

def preprocess_data(policies_df: pd.DataFrame, claims_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess the data.
    """
    policies_df = _light_feature_engineering_pandas(policies_df)
    claims_df = _light_feature_engineering_pandas(claims_df)

    claims_df[CAT_CODE] = claims_df[CAT_CODE].fillna("baseline")
    claims_df[CAT_CODE] = claims_df[CAT_CODE].replace("", "baseline")
    claims_df[CAT_CODE] = claims_df[CAT_CODE].str.strip()
    claims_df[CAT_CODE] = claims_df[CAT_CODE].str.lower()

    policies_df = policies_df.rename(columns={POLICY_SOLD_COUNT: POLICY_COUNT, 'TargetFileDescr': SEGMENT})
    claims_df = claims_df.rename(columns={CLAIM_NUM_UNIQUE: CLAIM_COUNT, 'TargetFileDescr': SEGMENT})

    
    return policies_df, claims_df

def compute_reported_frequency(
    policies_seg: pd.DataFrame, 
    claims_seg: pd.DataFrame,
    groupby_cols_policy: list[str] = [DEPARTURE_YEAR, PERIOD],
    groupby_cols_claim: list[str] = [DEPARTURE_YEAR, PERIOD, CAT_CODE]
   
) -> pd.DataFrame:
    """
    Compute the observed reported frequency by aggregating policies and claims by specified grouping columns.
    
    Args:
        policies_seg: DataFrame containing policy data for a segment
        claims_seg: DataFrame containing claim data for a segment
        groupby_cols_policy: List of column names to group by (default: [DEPARTURE_YEAR, PERIOD])
        groupby_cols_claim: List of column names to group by (default: [DEPARTURE_YEAR, PERIOD, CAT_CODE])
        
    Returns:
        DataFrame with reported frequency calculations with the following columns:
        - groupby_cols_policy
        - groupby_cols_claim
        - POLICY_COUNT
        - CLAIM_COUNT
        - FREQUENCY_VAL
    """
    # Aggregate policies
    policies_agg = policies_seg.groupby(groupby_cols_policy).agg({POLICY_COUNT: 'sum'}).reset_index()

    # Aggregate claims
    claims_agg = claims_seg.groupby(groupby_cols_claim).agg({CLAIM_COUNT: 'sum'}).reset_index()

    # Calculate reported frequency
    freq_df = pd.merge(policies_agg, claims_agg, on=groupby_cols_policy, how='left')
    freq_df[CLAIM_COUNT] = freq_df[CLAIM_COUNT].fillna(0)
    freq_df[FREQUENCY_VAL] = freq_df[CLAIM_COUNT] / freq_df[POLICY_COUNT]
    
    return freq_df


# if for a given segment x cat code the number of claims is above a threshold, then the cat code is a major cat
def flag_major_cats_pandas(df: pd.DataFrame, num_claims_threshold: int) -> pd.DataFrame:
    """
    Flag major cats based on the number of claims.
    If cat code is not baseline and above threshold, flag it as major cat, create a new column called is_major_cat.
    Group by segment and cat code to get the number of claims per cat code.
    """
    # Group by segment and cat code to get the number of claims per cat code
    df_grouped = (
        df.copy()
        .groupby([SEGMENT, CAT_CODE])
        .agg(claim_count_sum=(CLAIM_COUNT, 'sum'))
        .reset_index()
    )
    df_grouped[IS_MAJOR_CAT] = (df_grouped[CAT_CODE] != "baseline") & (df_grouped['claim_count_sum'] >= num_claims_threshold)

    # merge is_major_cat back to the original dataframe
    df = pd.merge(df, df_grouped[[SEGMENT, CAT_CODE, IS_MAJOR_CAT]], on=[SEGMENT, CAT_CODE], how='left')
    return df


def get_major_cat_events_table_too_complex(
    claims: pd.DataFrame,
    policies: pd.DataFrame,
    segment: str,
    small_cat_thresh: float = 0.01,
    abs_std_mult: float = 3.0,
    rel_ratio_thresh: float = 0.5
) -> pd.DataFrame:
    """
    Returns a DataFrame with major cat events for the given segment, including:
    - segment
    - catCode
    - event_start_date (earliest dateDepart among claims for that cat)
    - num_claims (number of claims for that cat)
    - frequency_val (claims for cat / policies for segment in depart year)
    - cat_severity (from flag_major_cats_pandas)
    """
    # Filter claims for the segment and with a cat code
    claims_seg = claims[(claims[SEGMENT] == segment) & claims[CAT_CODE].notna() & (claims[CAT_CODE] != "")].copy()
    if claims_seg.empty:
        return pd.DataFrame(columns=[SEGMENT, CAT_CODE, 'event_start_date', 'num_claims', FREQUENCY_VAL, 'cat_severity'])

    # Add departure year
    claims_seg[DEPARTURE_YEAR] = claims_seg[DEPARTURE_DAY].dt.year

    # Identify major cats
    flagged = flag_major_cats_pandas(claims_seg, small_cat_thresh, abs_std_mult, rel_ratio_thresh)
    # Only keep rows with a cat code
    flagged = flagged[flagged[CAT_CODE].notna() & (flagged[CAT_CODE] != "")]
    # For each cat, get is_major_cat and optionally more granularity
    flagged = flagged[[CAT_CODE, DEPARTURE_YEAR, 'is_major_cat']].drop_duplicates()

    # For each cat, get event_start_date and num_claims
    cat_summary = (
        claims_seg
        .groupby([CAT_CODE, DEPARTURE_YEAR])
        .agg(
            event_start_date=(DEPARTURE_DAY, 'min'),
            num_claims=(CAT_CODE, 'count')
        )
        .reset_index()
    )

    # Merge severity info
    cat_summary = pd.merge(cat_summary, flagged, on=[CAT_CODE, DEPARTURE_YEAR], how='left')
    cat_summary['cat_severity'] = cat_summary['is_major_cat'].map(lambda x: 'large' if x else 'small')

    # Add segment
    cat_summary[SEGMENT] = segment

    # Compute frequency_val: num_claims / total policies for segment in depart year
    policies_seg = policies[policies[SEGMENT] == segment].copy()
    policies_seg[DEPARTURE_YEAR] = policies_seg[DATE_DEPART_END_OF_MONTH].dt.year
    policy_counts = policies_seg.groupby(DEPARTURE_YEAR)[POLICY_COUNT].sum().reset_index()
    cat_summary = pd.merge(cat_summary, policy_counts, on=DEPARTURE_YEAR, how='left')
    cat_summary[FREQUENCY_VAL] = cat_summary['num_claims'] / cat_summary[POLICY_COUNT]

    # Final columns
    cat_summary = cat_summary[[SEGMENT, CAT_CODE, 'event_start_date', 'num_claims', FREQUENCY_VAL, 'cat_severity']]
    return cat_summary.sort_values('event_start_date')



def filter_and_group_by_period(
    policies_df: pd.DataFrame,
    claims_df: pd.DataFrame,
    segment: str,
    date_range: tuple,
    granularity: str,
    with_large_cat_events: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Filter data by segment and date range, then group by specified time period.
    
    Args:
        policies_df: DataFrame containing policy data
        claims_df: DataFrame containing claims data
        segment: Segment to filter by
        date_range: Tuple of (start_date, end_date)
        granularity: One of 'Month', 'Week', or 'Day'
        
    Returns:
        Tuple of (filtered_policies_df, filtered_claims_df, x_label)
    """

    #print(f"with_large_cat_events: {with_large_cat_events}")
    if with_large_cat_events == False:
        claims_seg = claims_df[~(claims_df[IS_MAJOR_CAT] == True)].copy()
    else:
        claims_seg = claims_df.copy()
    # Filter by segment and date range
    policies_seg = policies_df[
        (policies_df[SEGMENT] == segment) & 
        (pd.to_datetime(policies_df[DEPARTURE_DAY]) >= pd.to_datetime(date_range[0])) &
        (pd.to_datetime(policies_df[DEPARTURE_DAY]) <= pd.to_datetime(date_range[1]))
    ].copy()

    claims_seg = claims_seg[
        (claims_seg[SEGMENT] == segment) & 
        (pd.to_datetime(claims_seg[DEPARTURE_DAY]) >= pd.to_datetime(date_range[0])) &
        (pd.to_datetime(claims_seg[DEPARTURE_DAY]) <= pd.to_datetime(date_range[1])) 
    ].copy()


    # Extract year, month, week, day based on granularity
    if granularity == 'Month':
        policies_seg[PERIOD] = policies_seg[DEPARTURE_DAY].dt.month
        claims_seg[PERIOD] = claims_seg[DEPARTURE_DAY].dt.month
        x_label = 'Month'
    elif granularity == 'Week':
        policies_seg[PERIOD] = policies_seg[DEPARTURE_DAY].dt.isocalendar().week
        claims_seg[PERIOD] = claims_seg[DEPARTURE_DAY].dt.isocalendar().week
        x_label = 'Week'
    else:
        policies_seg[PERIOD] = policies_seg[DEPARTURE_DAY].dt.dayofyear
        claims_seg[PERIOD] = claims_seg[DEPARTURE_DAY].dt.dayofyear
        x_label = 'Day of Year'

    # Add departure year
    policies_seg[DEPARTURE_YEAR] = policies_seg[DEPARTURE_DAY].dt.year
    claims_seg[DEPARTURE_YEAR] = claims_seg[DEPARTURE_DAY].dt.year


    return policies_seg, claims_seg, x_label


def compute_development_factors_deprecated(freq_dev_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute development factors from the cumulative frequency of developed cohorts.
    A development factor is calculated as: freq_at_dev_d / freq_at_dev_{d-1}.
    The factor is associated with the 'from' development period (dev_{d-1}).

    Parameters:
    - freq_dev_df: DataFrame with columns [SEGMENT, DATE_DEPART_END_OF_MONTH, DEVELOPMENT, FREQUENCY_CUMUL].
                   DEVELOPMENT is expected to be an integer representing development periods (e.g., months).
    
    Returns:
    - DataFrame with development factors. Columns: 
      [SEGMENT, DATE_DEPART_END_OF_MONTH, DEVELOPMENT, DEVELOPMENT_FACTOR, FREQUENCY_CUMUL, PREV_FREQUENCY_CUMUL],
      where DEVELOPMENT is the 'from' period (integer).
      FREQUENCY_CUMUL is the cumulative frequency at the current development period.
      PREV_FREQUENCY_CUMUL is the cumulative frequency at the previous development period.
    """
    
    # Ensure a copy to avoid modifying the original DataFrame
    df = freq_dev_df.copy()

    # Sort by cohort and development period to ensure correct shift operation
    # This is crucial for the .shift() operation to work correctly within each group
    df.sort_values(by=[SEGMENT, DATE_DEPART_END_OF_MONTH, DEVELOPMENT], inplace=True)
    
    group_cols = [SEGMENT, DATE_DEPART_END_OF_MONTH]
    
    # Get the cumulative frequency and development period from the immediately preceding period within each group
    df['prev_freq_cumul'] = df.groupby(group_cols)[FREQUENCY_CUMUL].shift(1)
    df['prev_development'] = df.groupby(group_cols)[DEVELOPMENT].shift(1)
    
    # Filter out rows that don't have a preceding period (i.e., the first development period for each cohort).
    # These rows will have NaN for 'prev_freq_cumul' and 'prev_development' after the shift.
    factors_df = df.dropna(subset=['prev_freq_cumul', 'prev_development'])
    
    # Filter out rows where the previous cumulative frequency is zero or less.
    # Division by zero or non-positive values would lead to errors or meaningless factors.
    # Frequencies are typically non-negative.
    factors_df = factors_df[factors_df['prev_freq_cumul'] > 0]
    
    # Calculate the development factor
    # DEVELOPMENT_FACTOR = current_cumulative_frequency / previous_cumulative_frequency
    factors_df[DEVELOPMENT_FACTOR] = factors_df[FREQUENCY_CUMUL] / factors_df['prev_freq_cumul']
    
    # Prepare the result DataFrame with the required columns.
    # The 'DEVELOPMENT' column in the output refers to 'prev_development' (the 'from' period for the factor).
    result_df = factors_df[[
        SEGMENT, 
        DATE_DEPART_END_OF_MONTH, 
        'prev_development', 
        DEVELOPMENT_FACTOR,
        FREQUENCY_CUMUL, # Added for debugging
        'prev_freq_cumul' # Added for debugging
    ]].copy() # Use .copy() to ensure it's a new DataFrame and avoid SettingWithCopyWarning

    # Rename 'prev_development' to the standard 'DEVELOPMENT' column name.
    # Cast the 'DEVELOPMENT' column to integer type, as 'prev_development' might be float due to NaNs introduced by shift.
    result_df.rename(columns={'prev_development': DEVELOPMENT,
                              'prev_freq_cumul': 'PREV_FREQUENCY_CUMUL'}, inplace=True) # Added rename for prev_freq_cumul
    result_df[DEVELOPMENT] = result_df[DEVELOPMENT].astype(int)
    
    return result_df


def compute_development_factors(freq_dev_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute development factors from the cumulative frequency of developed cohorts.
    Returns DataFrame with development factors and their metadata.
    """
    df = freq_dev_df.copy()
    df.sort_values(by=[SEGMENT, DATE_DEPART_END_OF_MONTH, DEVELOPMENT], inplace=True)
    
    # Calculate previous period values
    df[PREV_FREQUENCY_CUMUL] = df.groupby([SEGMENT, DATE_DEPART_END_OF_MONTH])[FREQUENCY_CUMUL].shift(1)
    #df['prev_development'] = df.groupby([SEGMENT, DATE_DEPART_END_OF_MONTH])[DEVELOPMENT].shift(1)
    
    # Filter and calculate factors
    factors_df = df.dropna(subset=[PREV_FREQUENCY_CUMUL])
    factors_df = factors_df[factors_df[PREV_FREQUENCY_CUMUL] > 0]
    factors_df[DEVELOPMENT_FACTOR] = factors_df[FREQUENCY_CUMUL] / factors_df[PREV_FREQUENCY_CUMUL]
    
    # Prepare result
    result_df = factors_df[[
        SEGMENT, DATE_DEPART_END_OF_MONTH, DEVELOPMENT,
        DEVELOPMENT_FACTOR, FREQUENCY_CUMUL, PREV_FREQUENCY_CUMUL,POLICY_COUNT
    ]].copy()
    
   
    result_df[DEVELOPMENT] = result_df[DEVELOPMENT].astype(int)
    
    return result_df

def calculate_development_metrics(dev_factors_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Calculate all development metrics in one pipeline.
    Returns a dictionary containing all calculated metrics.
    """
    if dev_factors_df.empty:
        return {
            'raw_avg': pd.DataFrame(),
            'raw_cumulative': pd.DataFrame(),
            'vol_weighted_avg': pd.DataFrame(),
            'vol_weighted_cumulative': pd.DataFrame()
        }
    
    # Count the number of cohorts per development period
    dev_counts = dev_factors_df.groupby(DEVELOPMENT)[DATE_DEPART_END_OF_MONTH].nunique().reset_index(name='n_cohorts')
    # Keep only development periods with at least 10 cohorts
    valid_devs = dev_counts[dev_counts['n_cohorts'] >= 10][DEVELOPMENT]
    # Filter the main DataFrame
    filtered_df = dev_factors_df[dev_factors_df[DEVELOPMENT].isin(valid_devs)]
    # Compute the mean
    raw_avg = round(filtered_df.groupby(DEVELOPMENT)[DEVELOPMENT_FACTOR].mean(), 2).reset_index(name='y')
    raw_avg = raw_avg[raw_avg[DEVELOPMENT]<=24]
    
    # Calculate volume-weighted average factors
    vol_weighted = dev_factors_df.groupby(DEVELOPMENT).apply(
        lambda x: np.average(x[DEVELOPMENT_FACTOR], weights=x[POLICY_COUNT])
    ).reset_index()
    vol_weighted.columns = [DEVELOPMENT, 'y']
    vol_weighted = vol_weighted[vol_weighted[DEVELOPMENT]<=24]

    # Calculate cumulative patterns
    def calculate_cumulative(df):
        if df.empty:
            return pd.DataFrame(columns=[DEVELOPMENT, 'y'])
        factors = df['y'].to_numpy()
        cumulative = np.cumprod(factors[::-1])[::-1]
        result = pd.DataFrame({
            DEVELOPMENT: df[DEVELOPMENT].tolist(),
            'y': cumulative
        })
        # Add ultimate period
        result = pd.concat([
            result,
            pd.DataFrame([{
                DEVELOPMENT: df[DEVELOPMENT].max() + 1,
                'y': 1.0
            }])
        ], ignore_index=True)
        return result
    
    return {
        'raw_avg': raw_avg,
        'raw_cumulative': calculate_cumulative(raw_avg),
        'vol_weighted_avg': vol_weighted,
        'vol_weighted_cumulative': calculate_cumulative(vol_weighted)
    }


def project_cohort_development(
    freq_dev_df: pd.DataFrame,
    metrics: Dict[str, pd.DataFrame],
    cutoff_date: pd.Timestamp,
    use_volume_weighted: bool = True
) -> pd.DataFrame:
    """
    For each cohort, merge observed development with the expected pattern, forward-fill for mature, project for immature.
    Returns a DataFrame with columns: SEGMENT, DATE_DEPART_END_OF_MONTH, DEVELOPMENT, cumulative_frequency, expected_pattern, is_observed
    """
    if freq_dev_df.empty:
        return pd.DataFrame()

    # Choose pattern
    pattern_df = metrics['vol_weighted_avg'] if use_volume_weighted and not metrics['vol_weighted_avg'].empty else metrics['raw_avg']
    if pattern_df.empty:
        return pd.DataFrame()
    dev_factor_map = dict(zip(pattern_df[DEVELOPMENT], pattern_df['y']))
    max_pattern_dev = min(24,max(dev_factor_map.keys()))
    
    min_pattern_dev = min(dev_factor_map.keys())
    output_rows = []
    for (segment, cohort_date), group in freq_dev_df.groupby([SEGMENT, DATE_DEPART_END_OF_MONTH]):
        
        #max index
       
        group = group.sort_values(DEVELOPMENT)
        # Calculate months since departure for this cohort
        months_since_departure = (pd.to_datetime(cutoff_date).to_period('M') - pd.to_datetime(cohort_date).to_period('M')).n
        # Merge observed with pattern
        merged = pd.DataFrame({DEVELOPMENT: range(min_pattern_dev, max(months_since_departure, max_pattern_dev)+1)})
        merged = merged.merge(group[[DEVELOPMENT, FREQUENCY_CUMUL]], on=DEVELOPMENT, how='left')
        merged = merged.merge(pattern_df[[DEVELOPMENT, 'y']], on=DEVELOPMENT, how='left', suffixes=('', '_pattern'))
        merged.rename(columns={'y': 'expected_pattern'}, inplace=True)
        # Find last observed
        obs_mask = ~merged[FREQUENCY_CUMUL].isna()
        if obs_mask.any():
            last_obs_idx = merged[obs_mask].index.max()
            last_obs_cumul = merged.loc[last_obs_idx, FREQUENCY_CUMUL]
        else:
            last_obs_idx = -1
            last_obs_cumul = 0.0
        # Mature cohort: forward-fill all missing with last observed
        if months_since_departure > max_pattern_dev:
            merged[FREQUENCY_CUMUL] = merged[FREQUENCY_CUMUL].ffill()
            
            merged['is_observed'] = obs_mask
            # All periods up to max_pattern_dev
            merged = merged[merged[DEVELOPMENT] <= max_pattern_dev]
        else:
            # Immature: project forward from last observed up to months_since_departure
            max_idx = len(merged) - 1
            
                
           
            cumul = last_obs_cumul
            for idx in range(last_obs_idx+1, max_idx+1):
                current_dev = merged.loc[idx, DEVELOPMENT]
                #prev_dev = merged.loc[idx-1, DEVELOPMENT]
                factor = dev_factor_map.get(current_dev, 1.0)
                cumul = cumul * factor
                merged.at[idx, FREQUENCY_CUMUL] = cumul
            merged['is_observed'] = ~merged[FREQUENCY_CUMUL].isna() & obs_mask
            # Mark projected
            for idx in range(last_obs_idx+1, max_idx+1):
                merged.at[idx, 'is_observed'] = False
            # Only keep up to months_since_departure
            merged = merged[merged[DEVELOPMENT] <= max(months_since_departure, max_pattern_dev)]
        # Add cohort/segment/date info
        merged[SEGMENT] = segment
        merged[DATE_DEPART_END_OF_MONTH] = cohort_date
        output_rows.append(merged[[SEGMENT, DATE_DEPART_END_OF_MONTH, DEVELOPMENT, FREQUENCY_CUMUL, 'expected_pattern', 'is_observed']])
    return pd.concat(output_rows, ignore_index=True) if output_rows else pd.DataFrame()

# # create a function to develp the current observed cohort to ultimate
# # develop the current observed cohort to ultimate
# dev_triangle = compute_claim_development_triangle_pandas(claims_seg,groupby_cols=[SEGMENT, DATE_DEPART_END_OF_MONTH, DEVELOPMENT])




def estimate_ultimate_frequencies(projected_df: pd.DataFrame) -> pd.DataFrame:
    #get the last cumulative frequency

    idx = projected_df.groupby([SEGMENT, DATE_DEPART_END_OF_MONTH])[DEVELOPMENT].idxmax()
    result_ultimate = projected_df.loc[idx].reset_index(drop=True)
    result_ultimate = result_ultimate.rename(columns={FREQUENCY_CUMUL: ULTIMATE_FREQUENCY})

    idx_observed = projected_df[projected_df['is_observed'] == True].groupby([SEGMENT, DATE_DEPART_END_OF_MONTH])[DEVELOPMENT].idxmax()
    result_observed = projected_df.loc[idx_observed].reset_index(drop=True)
    result_observed = result_observed.rename(columns={FREQUENCY_CUMUL: CURRENT_FREQUENCY})

    # merge the result with the projected_df
    result = result_observed[[SEGMENT, DATE_DEPART_END_OF_MONTH,CURRENT_FREQUENCY]].merge(result_ultimate[[SEGMENT, DATE_DEPART_END_OF_MONTH,ULTIMATE_FREQUENCY]], on=[SEGMENT, DATE_DEPART_END_OF_MONTH], how='left')
    # calculate the ultimate frequency
    # get the departure month
    result[PERIOD] = result[DATE_DEPART_END_OF_MONTH].dt.month
    return result


# compute the raw average ultimate frequency for each development period using the estimated ultimate frequency
def compute_raw_average_ultimate_frequency(
    ultimate_freq: pd.DataFrame,
    exclude_date_ranges: list = None,  # List of (start, end) tuples or single months as strings (YYYY-MM)
    min_max_development: int = -24,
    cohort_dev_df: pd.DataFrame = None,  # Optionally, a DataFrame with cohort max development info,
    use_volume_weighted:bool = False
) -> pd.DataFrame:
    """
    Compute the raw average ultimate frequency, with options to exclude cohorts by date and by min max development.
    - exclude_date_ranges: list of (start, end) tuples or single months as 'YYYY-MM'
    - min_max_development: minimum max development to keep a cohort
    - cohort_dev_df: DataFrame with columns [SEGMENT, DATE_DEPART_END_OF_MONTH, max_development]
    """
    df = ultimate_freq.copy()
    # Exclude by date ranges
    if exclude_date_ranges:
        # Convert cohort date to YYYY-MM string
        df['cohort_month'] = df[DATE_DEPART_END_OF_MONTH].dt.strftime('%Y-%m')
        exclude_months = set()
        for item in exclude_date_ranges:
            if isinstance(item, tuple):
                # Range: ('2021-01', '2021-03')
                start, end = pd.to_datetime(item[0]), pd.to_datetime(item[1])
                months = pd.period_range(start, end, freq='M').strftime('%Y-%m').tolist()
                exclude_months.update(months)
            else:
                # Single month: '2021-08'
                exclude_months.add(item)
        df = df[~df['cohort_month'].isin(exclude_months)]
        df = df.drop(columns=['cohort_month'])
    # Exclude by min max development
    if cohort_dev_df is not None:
        # cohort_dev_df should have [SEGMENT, DATE_DEPART_END_OF_MONTH, max_development]
        df = df.merge(cohort_dev_df[[SEGMENT, DATE_DEPART_END_OF_MONTH,'max_development']],
                      on=[SEGMENT, DATE_DEPART_END_OF_MONTH], how='left')
        df = df[df['max_development'] >= min_max_development]
        df = df.drop(columns=['max_development'])
    #import streamlit as st
    #st.subheader("raw_avg_ultimate_freq")
    #st.dataframe(df)
    # Compute the mean
    def weighted_avg(group, value_col,weight_col):
        # only compute if there are non null wieghtes and values
        valid =group[weight_col].notnull() & group[value_col].notnull()
        if valid.any():
            return np.average(group.loc[valid,value_col], weights=group.loc[valid,weight_col])
        else:
            np.nan

    if use_volume_weighted == False:
        result_ultimate = df.groupby([SEGMENT, PERIOD]).agg({ULTIMATE_FREQUENCY: 'mean', CURRENT_FREQUENCY: 'mean'}).reset_index()
    else: # or true
        result_ultimate = df.groupby([SEGMENT, PERIOD]).apply(lambda g : pd.Series({
            ULTIMATE_FREQUENCY: weighted_avg(g,ULTIMATE_FREQUENCY,POLICY_COUNT),
            CURRENT_FREQUENCY: weighted_avg(g,CURRENT_FREQUENCY,POLICY_COUNT)
        })).reset_index()
    return result_ultimate.sort_values(by=[SEGMENT, PERIOD]),df


# --- Best frequency selection ---
def select_best_frequency(projected_df, ultimate_freq_avg, result_ultimate,min_max_development,lag):
    """
    For each cohort:
    - if the max development is greater than the expected development length, then use the reported frequency
    - otherwise, use the ultimate frequency
    Also forecast cohorts (year x period) up to December of the second full calendar year after the last observed cohort.
    """
    result_ultimate = result_ultimate.copy()
    last_date = pd.to_datetime(result_ultimate[DATE_DEPART_END_OF_MONTH].max())
    result_ultimate['year'] = result_ultimate[DATE_DEPART_END_OF_MONTH].dt.year
    result_ultimate['month'] = result_ultimate[DATE_DEPART_END_OF_MONTH].dt.month
    merged = result_ultimate.merge(
        ultimate_freq_avg[[SEGMENT, 'period', 'ultimate_frequency']],
        left_on=[SEGMENT, 'month'],
        right_on=[SEGMENT, 'period'],
        how='left'
    )
    expected_dev_length = projected_df[projected_df['is_observed'] == False][DEVELOPMENT].max()
    def best_freq(row):
        month_diff = (last_date.year - pd.to_datetime(row[DATE_DEPART_END_OF_MONTH]).year) * 12 + (last_date.month - pd.to_datetime(row[DATE_DEPART_END_OF_MONTH]).month)
        
        if (month_diff < lag) & (lag > 0):
            return row['ultimate_frequency']
        if (month_diff < min_max_development) & (lag ==0):
            return row['ultimate_frequency']
        if month_diff >= (5 + lag) : #cohort considered already developed
            return row[FREQUENCY_CUMUL]
        if month_diff >= (min_max_development) :
            return row[FREQUENCY_CUMUL]
        if month_diff >= (np.max([5,(min_max_development)])):
            return row[FREQUENCY_CUMUL]
        else:
            return max(row[FREQUENCY_CUMUL], row['ultimate_frequency'])
    merged['best_frequency'] = merged.apply(best_freq, axis=1)
    last_date = pd.to_datetime(result_ultimate[DATE_DEPART_END_OF_MONTH].max()) #+ pd.offsets.MonthEnd(-lag)
    last_year = last_date.year
    last_month = last_date.month
    segment_val = merged[SEGMENT].iloc[0] if not merged.empty else None
    final_year = last_year + 3
    forecast_months = []
    for m in range(last_month + 1, 13):
        forecast_months.append((last_year, m))
    for y in range(last_year + 1, final_year + 1):
        for m in range(1, 13):
            forecast_months.append((y, m))
    forecast_rows = []
    for y, m in forecast_months:
        uf_row = ultimate_freq_avg[(ultimate_freq_avg[SEGMENT] == segment_val) & (ultimate_freq_avg['period'] == m)]
        uf = uf_row['ultimate_frequency'].values[0] if not uf_row.empty else float('nan')
        forecast_rows.append({
            SEGMENT: segment_val,
            DATE_DEPART_END_OF_MONTH: pd.Timestamp(year=y, month=m, day=1),
            'year': y,
            'month': m,
            'max_development': float('nan'),
            FREQUENCY_CUMUL: float('nan'),
            'ultimate_frequency': uf,
            'best_frequency': uf
        })
    forecast_df = pd.DataFrame(forecast_rows)
    merged = pd.concat([merged, forecast_df], ignore_index=True)
    merged = merged.sort_values(['year', 'month']).reset_index(drop=True)
    
    # Step 1: Build full grid of year-month combinations
    first_date = pd.to_datetime(result_ultimate[DATE_DEPART_END_OF_MONTH].min())
    full_month_range = pd.date_range(
        start=first_date, 
        end=pd.Timestamp(year=final_year, month=12, day=1), 
        freq='MS'
    )

    full_grid = pd.DataFrame({
        DATE_DEPART_END_OF_MONTH: full_month_range
    })
    full_grid['year'] = full_grid[DATE_DEPART_END_OF_MONTH].dt.year
    full_grid['month'] = full_grid[DATE_DEPART_END_OF_MONTH].dt.month
    full_grid[SEGMENT] = segment_val  # Assumes 1 segment per call

    # Step 2: Merge with full grid to enforce completeness
    merged_full = full_grid.merge(
        merged,
        on=[SEGMENT, 'year', 'month', DATE_DEPART_END_OF_MONTH],
        how='left'
    )

    #filling missing ultimate frequencies
    merged_full = merged_full.merge(ultimate_freq_avg[[SEGMENT,'period', 'ultimate_frequency']]
                                    ,left_on=[SEGMENT, 'month'],
                                    right_on=[SEGMENT, 'period'],
                                    how='left'
                                    ,suffixes=('','_filled'))
    
    merged_full['ultimate_frequency'] = merged_full['ultimate_frequency'].combine_first(merged_full['ultimate_frequency_filled'])
    # Step 3: Recompute best_frequency if needed
    def recompute_freq(row):
        if pd.notna(row['best_frequency']):
            return row['best_frequency']
        if pd.isna(row['max_development']):
            return row['ultimate_frequency']
        if row['max_development'] < lag:
            return row['ultimate_frequency']
        if row['max_development'] >= (np.max([5, min_max_development])):
            return row[FREQUENCY_CUMUL]
        else:
            return max(row.get(FREQUENCY_CUMUL, 0), row.get('ultimate_frequency', 0))

    merged_full['best_frequency'] = merged_full.apply(recompute_freq, axis=1)
    return merged_full[[SEGMENT, DATE_DEPART_END_OF_MONTH, 'year', 'month', 'max_development', FREQUENCY_CUMUL, 'ultimate_frequency', 'best_frequency']].sort_values(['year','month']).reset_index(drop=True)

# def save_data(segment, cutoff_date, best_frequencies, results_path="_results"):
#     # Ensure results folder exists
#     if not os.path.exists(results_path):
#         os.makedirs(results_path)
#     cutoff_date = pd.to_datetime(cutoff_date)
#     #cutoff_str = cutoff_date #.strftime("%Y-%m-%d")

#     # Prepare best frequencies DataFrame
#     df_best = best_frequencies.copy()
#     df_best['segment'] = segment
#     df_best['cutoff'] = pd.to_datetime(cutoff_date)

#     # Save best frequencies
#     best_path = os.path.join(results_path, 'best_frequencies.csv')
#     if os.path.exists(best_path):
#         existing_best_df = pd.read_csv(best_path)
#         existing_best_df['cutoff'] = pd.to_datetime(existing_best_df['cutoff'])
#         mask = ~(
#             (existing_best_df['segment'] == segment) &
#             (existing_best_df['cutoff'] == pd.to_datetime(cutoff_date))
#         )
#         existing_best_df = existing_best_df[mask]
#         existing_best_df = pd.concat([existing_best_df, df_best], ignore_index=True)
#     else:
#         existing_best_df = df_best
#     existing_best_df.to_csv(best_path, index=False)



def _save_csv_no_duplicates(filepath, new_df, key_columns):
    """
    Append new_df to filepath, but remove any existing rows with the same key_columns as in new_df.
    """
    if os.path.exists(filepath):
        existing_df = pd.read_csv(filepath)
        existing_df['cutoff'] = pd.to_datetime(existing_df['cutoff'])
        # existing_df['cutoff_finance'] = pd.to_datetime(existing_df['cutoff_finance'])
        # existing_df['cutoff_frequency'] = pd.to_datetime(existing_df['cutoff_frequency'])

        # Remove rows that match any (segment, cutoff, cutoff_finance) in new_df
        mask = ~existing_df.set_index(key_columns).index.isin(new_df.set_index(key_columns).index)
        combined_df = pd.concat([existing_df[mask], new_df], ignore_index=True)
    else:
        combined_df = new_df
    combined_df.to_csv(filepath, index=False)


def save_data(segment, cutoff_date
              ,best_frequencies
              , results_path="_results"):
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
     # Prepare best frequencies DataFrame
    r1_df = best_frequencies.copy()

    r1_df[SEGMENT] = segment
    r1_df["cutoff"] = pd.to_datetime(cutoff_date)

    _save_csv_no_duplicates(
        os.path.join(results_path, 'best_frequencies.csv'),
        r1_df,
        [SEGMENT, 'cutoff']
    )

def exclude_dates_to_str(exclude_dates):
        items = []
        for item in exclude_dates:
            if isinstance(item, (tuple, list)) and len(item) == 2:
                items.append(f"{item[0]} to {item[1]}")
            else:
                items.append(str(item))
        return ", ".join(items)

def parse_exclude_dates(text):
        items = [x.strip() for x in text.split(",") if x.strip()]
        result = []
        for item in items:
            if "to" in item:
                start, end = [x.strip() for x in item.split("to")]
                result.append((start, end))
            else:
                result.append(item)
        return result

# --- Segment analysis and save function ---
def analyze_and_save_segment(segment
                             , cutoff_date
                             , all_configs
                             , policies_df
                             , claims_df
                             , date_range
                             , granularity
                             , with_large_cat_events
                             , config_path
                             , results_path="_results"
                             , lag=0):
    """Analyze a segment and save best frequencies, applying manual overrides if present."""
    
    use_volume_weighted = True # by default
    config_key = f"{segment}__{str(cutoff_date)}"
    user_config = all_configs.get(config_key, {})

    if 'manual_overrides' not in user_config:
            user_config['manual_overrides'] = []
    # min_max_dev = segment_config.get('min_max_dev', 0)
    min_max_development = user_config.get("min_max_development", 0)
    min_date = pd.to_datetime(user_config.get('min_date', policies_df[DEPARTURE_DAY].min()))
    max_date = pd.to_datetime(user_config.get('max_date', policies_df[DEPARTURE_DAY].max()))
    exclude_dates = exclude_dates_to_str(user_config.get('exclude_dates', []))

    exclude_dates_list = parse_exclude_dates(exclude_dates)

    num_claims_threshold = user_config.get("num_claims_threshold",100)
    claims_df_seg = flag_major_cats_pandas(claims_df,num_claims_threshold)

    policies_seg, claims_seg, x_label = filter_and_group_by_period(policies_df, claims_df_seg, segment, date_range, granularity, with_large_cat_events=False)
    
    
    # reported_freq_all_cats = compute_reported_frequency(policies_seg, claims_seg, groupby_cols_policy=[SEGMENT,DEPARTURE_YEAR, PERIOD], groupby_cols_claim=[SEGMENT,DEPARTURE_YEAR, PERIOD,CAT_CODE])
    # reported_freq_all_cats_agg = (
    #     reported_freq_all_cats.groupby([SEGMENT,DEPARTURE_YEAR, PERIOD])
    #         .agg({FREQUENCY_VAL: 'sum'})
    #         .reset_index()
    # )
    dev_triangle = compute_claim_development_triangle_pandas(claims_seg,groupby_cols=[SEGMENT, DATE_DEPART_END_OF_MONTH, DEVELOPMENT])
    dev_policies = policies_seg.groupby([SEGMENT, DATE_DEPART_END_OF_MONTH]).agg({POLICY_COUNT: 'sum'}).reset_index()
    freq_dev_cohort = compute_frequency_dev_pandas(dev_triangle, dev_policies, groupby_cols=[SEGMENT, DATE_DEPART_END_OF_MONTH])
    individual_dev_factors = compute_development_factors(freq_dev_cohort)
    metrics = calculate_development_metrics(individual_dev_factors)
    projected_df = project_cohort_development(freq_dev_cohort, metrics, pd.to_datetime(cutoff_date),use_volume_weighted=False)
    ultimate_freq = estimate_ultimate_frequencies(projected_df)
    ultimate_freq = ultimate_freq.merge(dev_policies[[SEGMENT,DATE_DEPART_END_OF_MONTH,POLICY_COUNT]],on=[SEGMENT,DATE_DEPART_END_OF_MONTH],how='left')

    idx = projected_df[projected_df['is_observed'] == True].groupby([SEGMENT, DATE_DEPART_END_OF_MONTH])[DEVELOPMENT].idxmax()
    result_ultimate = projected_df.loc[idx].reset_index().rename(columns={DEVELOPMENT: 'max_development'})
    
    idx = projected_df.groupby([SEGMENT, DATE_DEPART_END_OF_MONTH])[DEVELOPMENT].idxmax()
    result_ultimate_r = projected_df.loc[idx][[SEGMENT, DATE_DEPART_END_OF_MONTH,FREQUENCY_CUMUL]]
    result_ultimate = result_ultimate[[SEGMENT, DATE_DEPART_END_OF_MONTH,'max_development']].merge(result_ultimate_r,on=[SEGMENT, DATE_DEPART_END_OF_MONTH],how='left')
    result_ultimate[DEPARTURE_YEAR] = result_ultimate[DATE_DEPART_END_OF_MONTH].dt.year
    result_ultimate[PERIOD] = result_ultimate[DATE_DEPART_END_OF_MONTH].dt.month


    ultimate_freq_avg_no_filter,df_ultimate_test_no_filter = compute_raw_average_ultimate_frequency(ultimate_freq,exclude_date_ranges=None,min_max_development=-24,cohort_dev_df=result_ultimate,use_volume_weighted=use_volume_weighted)
    ultimate_freq_avg,df_ultimate_test_ = compute_raw_average_ultimate_frequency(ultimate_freq,exclude_dates_list,min_max_development,cohort_dev_df=result_ultimate,use_volume_weighted=use_volume_weighted)

    # result_ultimate[DEPARTURE_YEAR] = result_ultimate[DATE_DEPART_END_OF_MONTH].dt.year
    # result_ultimate[PERIOD] = result_ultimate[DATE_DEPART_END_OF_MONTH].dt.month
    # ultimate_freq_avg,df_ultimate_test_ = compute_raw_average_ultimate_frequency(ultimate_freq,exclude_dates_list,min_max_development,cohort_dev_df=result_ultimate)
    best_frequencies = select_best_frequency(projected_df, ultimate_freq_avg, result_ultimate,min_max_development=min_max_development,lag=lag)
    if 'source' not in best_frequencies.columns:
        best_frequencies['source'] = 'model'

    

    manual_overrides = get_manual_overrides(all_configs, config_key)
    if 'source' not in best_frequencies.columns:
        best_frequencies['source'] = 'model'
    for override in manual_overrides:
        mask = (best_frequencies['year'] == override['year']) & (best_frequencies['month'] == override['month'])
        best_frequencies.loc[mask, 'best_frequency'] = override['best_frequency']
        best_frequencies.loc[mask, 'source'] = 'manual'
    
    save_data(segment, cutoff_date, best_frequencies, results_path=results_path)



# block to import data for tripmate


def _list_csv_tripmate(folder_path:str):
    """ list the csv to buid the claims for tripmate pipeline """
    
    if os.path.exists(folder_path):
        csv_files_updated = [f"{folder_path}{file}" for file in os.listdir(folder_path) if file.endswith('.csv')]
        print(f"csv_uploaded:{csv_files_updated}")
    else:
        csv_files_updated = "The specified directory path does not exist."

    return csv_files_updated

def _create_raw_claims_flat_file_tripmate(folder_path:str):
    """ create the raw flat file for tripmate """
    # list the csv to import
    csv_tm = _list_csv_tripmate(folder_path)
    #encoding='iso-8859-1'
    df = pd.concat(
        [pd.read_csv(f,encoding_errors= 'ignore').assign(SOURCE=Path(f).name) for f in csv_tm],
    axis=0
    )
    #return the flat file
    return df


# # standardization of key dates
def _standardization(df, col):
    df[f'{col}_EndOfMonth'] = pd.to_datetime(df[col]).dt.to_period('M').dt.to_timestamp()
    df[f"{col}_month"] = df[f'{col}_EndOfMonth'].apply(lambda x: x.month)
    df[f"{col}_year"] = df[f'{col}_EndOfMonth'].apply(lambda x: x.year)  
    return df  

def _grouping_to_targetFileDescr(df):
    grouping_values = ["ALG","Fareportal","GroupCollect","OneTravel","Pleasant Holidays","Unique Vacations","Viking Cruises","World Nomads"]
    df['TargetFileDescr'] = df['grouping'].apply(lambda x: x if x in grouping_values else 'Other')
    return df


def _clean_claims_tripmate(raw_flat_df:pd.DataFrame,valuation_date:str,inception_date:str=None,apply_filter:bool=True):

    # clean the dates
    dates = ["INCR DATE ACT"
            ,"PREM RCVD DATE"
            ,"PremiumRecdMonth"
            ,"ReportDateMonth"
            ,"DepartDateMonth"]
    for col in dates:
        raw_flat_df[col] = pd.to_datetime(raw_flat_df[col], errors="coerce")

    
    # # rename columns
    pivot = raw_flat_df.rename(columns={"Carrier":"accountCarrier"
                                    ,"SelGrouping":"grouping"
                                    ,"UniqueClaim":"clmNum_unique"
                                    ,"Plan_Number":"plan_number"
                                    ,"DepartDateMonth":"dateDepart"
                                    ,"PremiumRecdMonth":"dateApp"
                                    ,"ReportDateMonth":"dateReceived"
                                    ,"INCR DATE ACT":"incrDateAct"})
   
    #return pivot
    

   # filter dates
    if apply_filter:
        incurred_after_start = (pivot["incrDateAct"]<=valuation_date)
        if inception_date is not None:
            incurred_before_valuation = (pivot["incrDateAct"]>=inception_date)
            pivot_clean = pivot[incurred_after_start & incurred_before_valuation]
        else:
            pivot_clean = pivot[incurred_after_start]
    else:
        pivot_clean = pivot
#     return pivot_clean
    # aggregate
    cols = [
        "PartAIndicator"
        ,"grouping"
        ,"accountCarrier"
        ,"SelPlanName"
        ,"plan_number"
        ,'incrDateAct'
        ,"PREM RCVD DATE"
        ,"PRM YR"
        ,"UWYear"
        ,"dateApp"
        ,"dateDepart"
        ,"dateReceived"        
        ]
    dict_agg = {"CLAIM num":"count"
                ,"PAIDCALCULATED":"sum"
                ,"RESERVECALCULATED":"sum"
                ,"INCURREDCALCULATED":"sum"
                ,"PAIDCOUNT":"sum"
                ,"NetPaid":"sum"
                ,"NetReserve":"sum"
                ,"NetIncurred":"sum"
                ,"PartAIndicator":"sum"
                ,"clmNum_unique":"sum"
            }

    agg_tm = pivot_clean.groupby(cols,as_index=False).agg(dict_agg)

    
    
    for col in ['dateApp','dateDepart','dateReceived']:
        agg_tm = _standardization(agg_tm,col)

    
    # aggregate grouping and standardize the column name:
        # Add a new column 'partnername' based on the logic
    agg_tm = _grouping_to_targetFileDescr(agg_tm)

    agg_tm['plan_number'] = agg_tm['plan_number'].str.upper()
    
    return agg_tm

def _main_claims_tripmate_csv_pipeline(folder_path:str,end_date:str,inception_date:str=None,appply_filter:bool=True):
    raw = _create_raw_claims_flat_file_tripmate(folder_path=folder_path)
    clean = _clean_claims_tripmate(raw_flat_df = raw
                                   ,valuation_date=end_date
                                   ,inception_date=inception_date
                                   ,apply_filter=appply_filter)
    return clean


# map plan to grouping from claims
def _map_plan_to_grouping(mapping_df:pd.DataFrame,df:pd.DataFrame)->pd.DataFrame:
    
    mapping_df['plan_number'] = mapping_df['plan_number'].str.upper()

    map_plan_to_grouping = df[["plan_number","grouping"]].drop_duplicates()
    # Merging the dataframes on 'plan_number'
    merged_df = map_plan_to_grouping.merge(mapping_df, on='plan_number', how='outer', suffixes=('_claim', '_upload'))

    # Selecting the 'grouping' from the second dataframe when they differ, else use the first
    merged_df['grouping'] = merged_df['grouping_upload'].fillna(merged_df['grouping_claim'])

    # Dropping the intermediate columns
    final_df = merged_df.drop(columns=['grouping_claim', 'grouping_upload'])
    return final_df

def load_updated_mapping_tm(path_mapping:str):
    return pd.read_csv(path_mapping)

def _find_inactive_plan_numbers_proxy(df:pd.DataFrame)->list:
    track_inactive_plans_proxy_1 = df[(df["dateDepart_year"].isin([2023,2024]))].groupby(["TargetFileDescr","plan_number"],as_index=False,dropna=False).agg({"idpol_unique":"sum","clmNum_unique":"sum"})
    track_inactive_plans_proxy_2 = df.groupby(["TargetFileDescr","plan_number"],as_index=False,dropna=False).agg({"dateDepart_year":"max"})

    track_inactive_plans_proxy_1 = track_inactive_plans_proxy_1[(track_inactive_plans_proxy_1["idpol_unique"]+track_inactive_plans_proxy_1["clmNum_unique"])<=100]
    track_inactive_plans_proxy_2 = track_inactive_plans_proxy_2[track_inactive_plans_proxy_2["dateDepart_year"]<=2022]

    unique_plan_numbers_1 = set(track_inactive_plans_proxy_1['plan_number'].unique())
    unique_plan_numbers_2 = set(track_inactive_plans_proxy_2['plan_number'].unique())


    # Use set union to get all unique 'plan_number' values from both sets
    all_unique_plan_numbers = unique_plan_numbers_1.union(unique_plan_numbers_2)

    # Convert the set back to a list if needed
    all_unique_plan_numbers_list = sorted(list(all_unique_plan_numbers))

    return all_unique_plan_numbers_list


def _clean_pol_tripmate(df_viking, df_not_viking, map_plan_to_grouping_df,clean_claims_df)->pd.DataFrame:
    
    cols = ["grouping","plan_number","dateApp","dateDepart"
            ,"dateApp_EndOfMonth","dateApp_month","dateApp_year"
            ,"dateDepart_EndOfMonth","dateDepart_month","dateDepart_year"
            ,"idpol_unique"]
    
    cols_claims = ["grouping","plan_number","dateApp","dateDepart","dateReceived"
            ,"dateApp_EndOfMonth","dateApp_month","dateApp_year"
            ,"dateDepart_EndOfMonth","dateDepart_month","dateDepart_year"
            ,"dateReceived_EndOfMonth","dateReceived_month","dateReceived_year"
            ]
    
    df_viking = df_viking[cols]

    # df pol not viking
    df_pol_not_viking = pd.merge(df_not_viking,map_plan_to_grouping_df,how="left",on=["plan_number"])
    df_pol_not_viking = df_pol_not_viking[cols]
    df_pol_not_viking = df_pol_not_viking[~(df_pol_not_viking["grouping"]=="Viking Cruises")]

    agg_clean_claims = clean_claims_df.groupby(cols_claims,as_index=False).agg({"clmNum_unique":"sum"})
    
    agg_ = pd.concat([df_pol_not_viking,df_viking,agg_clean_claims],axis=0)

    agg_ = _grouping_to_targetFileDescr(agg_)
    
    agg_["app_to_depart_month"] = agg_.apply(lambda row: calculate_month_difference("dateApp", "dateDepart", row), axis=1)

    agg_["depart_to_receive_month"] = agg_.apply(lambda row: calculate_month_difference("dateDepart", "dateReceived", row), axis=1)

    return agg_


def load_data_backup_tripmate(cutoff_date:str,backup_root:str):
        """
        In the case the current lakehouse is down, rollback to .csv files extracted using MISDB / legacy system.
        """
        # backup_root = "C:\\Users\\sfurderer\\OneLake - Microsoft\\USTI-ACTUARIAL-DEV\\USTI_IDEA_SILVER.Lakehouse\\Files\\_data\\tripmate\\"
        claims_backup_folder = backup_root + cutoff_date + "\\claims\\"
        policies_backup_folder = backup_root + cutoff_date + "\\policies\\"

        claims_df_PATH = f"{backup_root}{cutoff_date}\\claims.parquet"
        mapping_PATH = f"{backup_root}{cutoff_date}\\mapping.parquet"
        non_viking_PATH = f"{backup_root}{cutoff_date}\\policies_not_viking.parquet"
        viking_PATH = f"{backup_root}{cutoff_date}\\policies_viking.parquet"
        preprocessed_PATH = f"{backup_root}{cutoff_date}\\preprocessed_df.parquet"
        claims_df_final_PATH = f"{backup_root}{cutoff_date}\\claims_final.parquet"
        policies_df_final_PATH = f"{backup_root}{cutoff_date}\\policies_final.parquet"

        mapping_file = backup_root + "mapping_tm.csv"
        mapping_df = pd.read_csv(mapping_file)

        # import raw claims
        print(f"claims_df_PATH:{claims_df_PATH}")
        if os.path.exists(claims_df_PATH):
            claims_df  = pd.read_parquet(claims_df_PATH)
        else:
            # read the file
            claims_df = _main_claims_tripmate_csv_pipeline(folder_path=claims_backup_folder
                                                             ,end_date=cutoff_date
                                                             ,inception_date='2018-01-01'
                                                             ,appply_filter=False)
            
            claims_df.to_parquet(claims_df_PATH)
        
        # create claims with segment
        if os.path.exists(mapping_PATH):
            claims_df_with_mapping  = pd.read_parquet(mapping_PATH)
        else:
            claims_df_with_mapping = _map_plan_to_grouping(mapping_df,claims_df)
            claims_df_with_mapping.to_parquet(mapping_PATH)
            # policies_file = backup_root + cutoff_date + "\\policies.csv"
        
        # import policies without Vikings
        
        if os.path.exists(non_viking_PATH):
            df_pol_not_viking = pd.read_parquet(non_viking_PATH)
        else:
            df_pol_not_viking = pd.read_csv(f"{policies_backup_folder}non_vikings.csv")
            df_pol_not_viking = df_pol_not_viking.rename(columns={
                                        "dateDepart_month":"dateDepart"
                                        ,"dateApp_month":"dateApp"
                                        ,"policy_sold_count":"idpol_unique"
                                        })
            
            df_pol_not_viking = df_pol_not_viking.groupby(["plan_number","dateApp","dateDepart"],as_index=False).agg({"idpol_unique":"sum"})

            for col in ["dateDepart","dateApp"]:
                df_pol_not_viking[col] = pd.to_datetime(df_pol_not_viking[col], errors="coerce")
                df_pol_not_viking = _standardization(df_pol_not_viking,col)

            df_pol_not_viking['plan_number'] = df_pol_not_viking['plan_number'].str.upper()
            df_pol_not_viking.to_parquet(non_viking_PATH)

        if os.path.exists(viking_PATH):
            df_pol_viking = pd.read_parquet(viking_PATH)
        else:
            df_pol_viking = pd.read_csv(f"{policies_backup_folder}vikings.csv")
            df_pol_viking["grouping"] = "Viking Cruises"
            df_pol_viking["plan_number"] = "NULL"
            df_pol_viking = df_pol_viking.rename(columns={
                                        "dateDepart_month":"dateDepart"
                                        ,"dateApp_month":"dateApp"
                                        ,"policy_sold_count":"idpol_unique"
                                        })
            
            for col in ["dateDepart","dateApp"]:
                df_pol_viking[col] = pd.to_datetime(df_pol_viking[col], errors="coerce")
                df_pol_viking = _standardization(df_pol_viking,col)
            
            df_pol_viking.to_parquet(viking_PATH)

        if os.path.exists(preprocessed_PATH):
            preprocessed_df = pd.read_parquet(preprocessed_PATH)
        else:
            preprocessed_df = _clean_pol_tripmate(df_pol_viking,df_pol_not_viking,claims_df_with_mapping,claims_df)
            inactive_plan_numbers = _find_inactive_plan_numbers_proxy(preprocessed_df)
            # remove them
            preprocessed_df = preprocessed_df[~(preprocessed_df["plan_number"].isin(inactive_plan_numbers))]

            condition_1 = preprocessed_df['dateApp']>='2018-01-01'
        # departure date after purchase date (remove inconsistency)
            condition_2 = preprocessed_df['dateDepart']>=preprocessed_df['dateApp']
            # remove the policy purchased before a certain time
            condition_3 = preprocessed_df['dateApp']<=cutoff_date
            # remove policies with a departure date beyon the forecasted date
            forecast_date = pd.to_datetime(cutoff_date) + relativedelta(years = 3)
            
            condition_4 = preprocessed_df['dateDepart']<= forecast_date
            # remove claims received after extraction_date
            #condition_5 = df_p['dateReceived'] <= extraction_date
            preprocessed_df = preprocessed_df[(condition_1) & (condition_2) & (condition_3) & (condition_4)]# 
        
            preprocessed_df.to_parquet(preprocessed_PATH)


        if os.path.exists(claims_df_final_PATH):
            claims_df_final = pd.read_parquet(claims_df_final_PATH)
        else:  
            claims_df_final = preprocessed_df[["TargetFileDescr","dateDepart","dateReceived","dateDepart_EndOfMonth","dateReceived_EndOfMonth","depart_to_receive_month","clmNum_unique"]].dropna()
            claims_df_final[CAT_CODE] = 'baseline'
            claims_df_final = claims_df_final.groupby(["TargetFileDescr",CAT_CODE,"dateDepart","dateReceived","dateDepart_EndOfMonth","dateReceived_EndOfMonth","depart_to_receive_month"]).agg({"clmNum_unique":"sum"}).reset_index()
            claims_df_final.to_parquet(claims_df_final_PATH)

        if os.path.exists(policies_df_final_PATH):
            policies_df_final = pd.read_parquet(policies_df_final_PATH)
        else:    
            policies_df_final = preprocessed_df[["TargetFileDescr","dateApp","dateDepart","dateApp_EndOfMonth","dateDepart_EndOfMonth","app_to_depart_month","idpol_unique"]].dropna()
            policies_df_final = policies_df_final.groupby(["TargetFileDescr","dateApp","dateDepart","dateApp_EndOfMonth","dateDepart_EndOfMonth","app_to_depart_month"]).agg({"idpol_unique":"sum"}).reset_index()
            policies_df_final.to_parquet(policies_df_final_PATH)
        
        policies_df_final = _light_feature_engineering_pandas_tripmate(policies_df_final)
        claims_df_final = _light_feature_engineering_pandas_tripmate(claims_df_final)
        
        policies_df_final = policies_df_final.rename(columns={"idpol_unique": POLICY_COUNT, 'TargetFileDescr': SEGMENT})
        claims_df_final = claims_df_final.rename(columns={"clmNum_unique": CLAIM_COUNT, 'TargetFileDescr': SEGMENT})
        
        return policies_df_final,claims_df_final


def load_data_backup(cutoff_date:str,backup_root:str):
    """
    In the case the current lakehouse is down, rollback to .csv files extracted using MISDB / legacy system.
    """
    #backup_root = "C:\\Users\\sfurderer\\OneLake - Microsoft\\USTI-ACTUARIAL-DEV\\USTI_IDEA_SILVER.Lakehouse\\Files\\_data\\csa\\"
    
    if os.path.exists(f"{backup_root}{cutoff_date}\\policies.parquet"):
        policies_df = pd.read_parquet(f"{backup_root}{cutoff_date}\\policies.parquet")
        claims_df  = pd.read_parquet(f"{backup_root}{cutoff_date}\\claims.parquet")
    else:
        claims_file = backup_root + cutoff_date + "\\claims.csv"
        policies_file = backup_root + cutoff_date + "\\policies.csv"
    
        claims_df = pd.read_csv(claims_file)
        policies_df = pd.read_csv(policies_file)

        def _standardization(df, col):
            df[f'{col}_EndOfMonth'] = pd.to_datetime(df[col]).dt.to_period('M').dt.to_timestamp()
            df[f"{col}_month"] = df[f'{col}_EndOfMonth'].apply(lambda x: x.month)
            df[f"{col}_year"] = df[f'{col}_EndOfMonth'].apply(lambda x: x.year)  
            return df  

        for col in [SOLD_DAY,DEPARTURE_DAY]:
            policies_df =_standardization(policies_df,col)
        
        for col in [SOLD_DAY,DEPARTURE_DAY,RECEIVED_DAY]:
            claims_df =_standardization(claims_df,col)

        for col in [SOLD_DAY,DEPARTURE_DAY,DATE_SOLD_END_OF_MONTH,DATE_DEPART_END_OF_MONTH]:
            policies_df[col] = pd.to_datetime(policies_df[col])
        
        for col in [SOLD_DAY,DEPARTURE_DAY,RECEIVED_DAY,DATE_SOLD_END_OF_MONTH,DATE_DEPART_END_OF_MONTH,DATE_RECEIVED_END_OF_MONTH]:
            claims_df[col] = pd.to_datetime(claims_df[col])
        
        policies_df , claims_df = preprocess_data(policies_df,claims_df)

        policies_df.to_parquet(f"{backup_root}{cutoff_date}\\policies.parquet")
        claims_df.to_parquet(f"{backup_root}{cutoff_date}\\claims.parquet")
    # write dataframe
    return policies_df , claims_df

__all__ = [
    'analyze_and_save_segment',
    'calculate_development_metrics',
    'compute_claim_development_triangle_pandas',
    'compute_development_factors',
    'compute_frequency_dev_pandas',
    'compute_raw_average_ultimate_frequency',
    'compute_reported_frequency',
    'estimate_ultimate_frequencies',
    'filter_and_group_by_period',
    'flag_major_cats_pandas',
    'get_manual_overrides',
    'load_data',
    'load_data_backup',
    'load_data_backup_tripmate',
    'load_user_config',
    'project_cohort_development',
    'save_data',
    'save_user_config',
    'select_best_frequency',
    'set_manual_overrides',
]


