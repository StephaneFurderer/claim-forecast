import pandas as pd
from frequency_development.constants import *

def calc_yoy(df, year, value_col, date_col, segment_col=SEGMENT):
    prev_year = year - 1
    df[date_col] = pd.to_datetime(df[date_col])
    # If DataFrame is already filtered by segment, don't group
    if len(df[segment_col].unique()) == 1:
        curr = df[(df[date_col].dt.year == year)][value_col].sum()
        prev = df[(df[date_col].dt.year == prev_year)][value_col].sum()
        yoy = ((curr / prev - 1) * 100) if prev else float('nan')
        return yoy, curr, prev
    else:
        # Group by segment and calculate YoY for each
        curr = df[(df[date_col].dt.year == year)].groupby(segment_col)[value_col].sum()
        prev = df[(df[date_col].dt.year == prev_year)].groupby(segment_col)[value_col].sum()
        yoy = ((curr / prev - 1) * 100).fillna(float('nan'))
        return yoy, curr, prev

def prepare_segment_monthly_data(df, segment, date_col, value_col, scenario_name, selected_year):
    """Prepare monthly aggregated data for a specific segment and scenario"""
    seg_data = df[df[SEGMENT] == segment]
    monthly = seg_data.groupby(date_col)[value_col].sum().reset_index()
    monthly['scenario'] = scenario_name
    monthly['year'] = monthly[date_col].dt.year
    monthly['month'] = monthly[date_col].dt.month
    monthly['scenario_year'] = monthly['scenario'] + '_' + monthly['year'].astype(str)
    monthly = monthly[monthly['year'].isin([selected_year])]
    return monthly

def get_segment_metrics(yoy_summary_df_selected_year, segment):
    """Extract key metrics for a specific segment"""
    seg_data = yoy_summary_df_selected_year[yoy_summary_df_selected_year[SEGMENT] == segment]
    if len(seg_data) == 0:
        return None
    
    return {
        'yoy_baseline': seg_data['YoY prev (baseline)'].values[0],
        'yoy_current': seg_data['YoY current (current)'].values[0],
        'diff_policy_volume_yoy': seg_data['Difference Policy Volume YoY'].values[0]
    }

def _agg_claims(df:pd.DataFrame, selected_year: int,col:str)-> pd.DataFrame:
    if selected_year is None:
        df_per_seg = df.groupby([SEGMENT,col]).agg({CLAIM_COUNT:"sum"}).reset_index().sort_values(by=[SEGMENT,col])
    else:   
        df_per_seg = df[df[col].dt.year == selected_year].groupby([SEGMENT,col]).agg({CLAIM_COUNT:"sum"}).reset_index().sort_values(by=[SEGMENT,col])
    df_per_month = df_per_seg.groupby(col).agg({CLAIM_COUNT:"sum"}).reset_index().sort_values(by=[col])
    df_total = df_per_seg[CLAIM_COUNT].sum()
    return df_per_seg, df_per_month,df_total

def agg_claims_per_dep_date(df:pd.DataFrame, selected_year: int)-> pd.DataFrame:
    return _agg_claims(df, selected_year,DATE_DEPART_END_OF_MONTH)

def agg_claims_received(df:pd.DataFrame, selected_year: int)-> pd.DataFrame:
    return _agg_claims(df, selected_year,DATE_RECEIVED_END_OF_MONTH)

def analyze_aoc_data(
    policies_base, policies_shift, policies_aoc,
    claims_base, claims_shift, claims_aoc,
    claims_base_received, claims_shift_received,claims_aoc_received,
    selected_year, cutoff_date
):
    segments = sorted(list(set(policies_base[SEGMENT].unique())-set(['Booking.com'])))
    
    # Calculate YoY for each scenario
    yoy_summary_rows = []
    years = [2023, 2024, 2025, 2026]

    for segment in segments:
       for year in years:
        # Calculate baseline YoY
        prev_year = year - 1
        base_yoy, base_curr, base_prev = calc_yoy(
            policies_base[policies_base[SEGMENT] == segment], 
            year, 
            POLICY_COUNT, 
            DATE_DEPART_END_OF_MONTH
            )
        shift_yoy, shift_curr, shift_prev = calc_yoy(
            policies_shift[policies_shift[SEGMENT] == segment], 
            year, 
            POLICY_COUNT, 
            DATE_DEPART_END_OF_MONTH
            )
    
        # Calculate frequency rates
        base_claims_val = claims_base[(claims_base[SEGMENT] == segment) & (claims_base[DATE_DEPART_END_OF_MONTH].dt.year == year)][CLAIM_COUNT].sum()
        shift_claims_val = claims_shift[(claims_shift[SEGMENT] == segment) & (claims_shift[DATE_DEPART_END_OF_MONTH].dt.year == year)][CLAIM_COUNT].sum()
    
        base_freq = (base_claims_val / base_curr) if base_curr else float('nan')
        shift_freq = (shift_claims_val / shift_curr ) if shift_curr else float('nan')
        freq_diff = shift_freq - base_freq
    
        # Get all months for the current year
        months_in_year = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='MS')

        # Observed period: months up to and including cutoff
        observed_months = months_in_year[months_in_year <= pd.Timestamp(cutoff_date)]
        forecasted_months = months_in_year[months_in_year > pd.Timestamp(cutoff_date)]

        # Baseline and shifted claims for observed period
        base_claims_observed = claims_base[
        (claims_base[SEGMENT] == segment) &
        (claims_base[DATE_DEPART_END_OF_MONTH].dt.year == year) &
        (claims_base[DATE_DEPART_END_OF_MONTH].isin(observed_months))
        ][CLAIM_COUNT].sum()

        shift_claims_observed = claims_shift[
        (claims_shift[SEGMENT] == segment) &
        (claims_shift[DATE_DEPART_END_OF_MONTH].dt.year == year) &
        (claims_shift[DATE_DEPART_END_OF_MONTH].isin(observed_months))
        ][CLAIM_COUNT].sum()

        # Baseline and shifted claims for forecasted period
        base_claims_forecast = claims_base[
        (claims_base[SEGMENT] == segment) &
        (claims_base[DATE_DEPART_END_OF_MONTH].dt.year == year) &
        (claims_base[DATE_DEPART_END_OF_MONTH].isin(forecasted_months))
        ][CLAIM_COUNT].sum()

        shift_claims_forecast = claims_shift[
        (claims_shift[SEGMENT] == segment) &
        (claims_shift[DATE_DEPART_END_OF_MONTH].dt.year == year) &
        (claims_shift[DATE_DEPART_END_OF_MONTH].isin(forecasted_months))
        ][CLAIM_COUNT].sum()

        # Differences
        diff_observed = shift_claims_observed - base_claims_observed
        diff_forecast = shift_claims_forecast - base_claims_forecast
        total_diff = diff_observed + diff_forecast
        
        yoy_summary_rows.append({
            SEGMENT: segment,
            'Year': year,
            'YoY prev (baseline)': base_yoy,
            'YoY current (current)': shift_yoy,
            'Difference Policy Volume YoY': (shift_yoy - base_yoy),
            'Policy Volume prev': int(base_prev),
            'Policy Volume current': int(base_curr),
            'Policy Volume prev (current)': int(shift_prev),
            'Policy Volume current (current)': int(shift_curr),
            'Claim Volume prev': int(base_claims_val),
            'Claim Volume current': int(shift_claims_val),
            'Difference Claim Volume': (shift_claims_val - base_claims_val),
            'Difference Claim Volume Observed': diff_observed,
            'Difference Claim Volume Forecast': diff_forecast,
            'Difference Claim Volume Total': total_diff,
            'Frequency Rate (baseline)': base_freq,
            'Frequency Rate (current)': shift_freq,
            'Frequency Rate Diff': freq_diff
            })

    yoy_summary_df = pd.DataFrame(yoy_summary_rows)

    def _aggregagate_per_dep_month(policies, claims):
        _pol = policies.groupby([SEGMENT,DATE_DEPART_END_OF_MONTH])[POLICY_COUNT].sum().reset_index()
        _pol = _pol.rename(columns={POLICY_COUNT: 'policy_volume'})

        _claims = claims.groupby([SEGMENT,DATE_DEPART_END_OF_MONTH])[CLAIM_COUNT].sum().reset_index()
        _claims = _claims.rename(columns={CLAIM_COUNT: 'claim_volume'})
    
        _merge = pd.merge(_pol, _claims, on=[SEGMENT,DATE_DEPART_END_OF_MONTH])
        _merge = _merge.sort_values([SEGMENT,DATE_DEPART_END_OF_MONTH])
        _merge['frequency'] = _merge['claim_volume'] / _merge['policy_volume']
        return _merge

    
    # Aggregate for baseline
    base_merged = _aggregagate_per_dep_month(policies_base,claims_base)
    # base_df = policies_base[policies_base[DATE_DEPART_END_OF_MONTH].dt.year == selected_year].groupby(SEGMENT)[POLICY_COUNT].sum().reset_index()
    # base_df = base_df.rename(columns={POLICY_COUNT: 'policy_volume'})
    # base_claims_agg = claims_base[claims_base[DATE_DEPART_END_OF_MONTH].dt.year == selected_year].groupby(SEGMENT)[CLAIM_COUNT].sum().reset_index()
    # base_claims_agg = base_claims_agg.rename(columns={CLAIM_COUNT: 'claim_volume'})
    # base_merged = pd.merge(base_df, base_claims_agg, on=SEGMENT)
    # base_merged['frequency'] = base_merged['claim_volume'] / base_merged['policy_volume']

    # Aggregate for shifted
    shift_merged = _aggregagate_per_dep_month(policies_shift,claims_shift)

    # shift_df = policies_shift[policies_shift[DATE_DEPART_END_OF_MONTH].dt.year == selected_year].groupby(SEGMENT)[POLICY_COUNT].sum().reset_index()
    # shift_df = shift_df.rename(columns={POLICY_COUNT: 'policy_volume'})
    # shift_claims_agg = claims_shift[claims_shift[DATE_DEPART_END_OF_MONTH].dt.year == selected_year].groupby(SEGMENT)[CLAIM_COUNT].sum().reset_index()
    # shift_claims_agg = shift_claims_agg.rename(columns={CLAIM_COUNT: 'claim_volume'})
    # shift_merged = pd.merge(shift_df, shift_claims_agg, on=SEGMENT)
    # shift_merged['frequency'] = shift_merged['claim_volume'] / shift_merged['policy_volume']

    aoc_merged = _aggregagate_per_dep_month(policies_aoc,claims_aoc)

    # aoc_df = policies_aoc[policies_aoc[DATE_DEPART_END_OF_MONTH].dt.year == selected_year].groupby(SEGMENT)[POLICY_COUNT].sum().reset_index()
    # aoc_df = aoc_df.rename(columns={POLICY_COUNT: 'policy_volume'})
    # aoc_claims_agg = claims_aoc[claims_aoc[DATE_DEPART_END_OF_MONTH].dt.year == selected_year].groupby(SEGMENT)[CLAIM_COUNT].sum().reset_index()
    # aoc_claims_agg = aoc_claims_agg.rename(columns={CLAIM_COUNT: 'claim_volume'})
    # aoc_merged = pd.merge(aoc_df, aoc_claims_agg, on=SEGMENT)
    # aoc_merged['frequency'] = aoc_merged['claim_volume'] / aoc_merged['policy_volume']

    base_claims_received_per_seg, base_claims_received_per_month, base_claims_received_total = agg_claims_received(claims_base_received, selected_year=None)
    aoc_claims_received_per_seg, aoc_claims_received_per_month, aoc_claims_received_total = agg_claims_received(claims_aoc_received, selected_year=None)
    shift_claims_received_per_seg, shift_claims_received_per_month, shift_claims_received_total = agg_claims_received(claims_shift_received,selected_year=None)

    base_claims_per_dep_per_seg, base_claims_per_dep_per_month, base_claims_per_dep_total = agg_claims_per_dep_date(claims_base, selected_year=None)
    aoc_claims_per_dep_per_seg, aoc_claims_per_dep_per_month, aoc_claims_per_dep_total = agg_claims_per_dep_date(claims_aoc, selected_year=None)
    shift_claims_per_dep_per_seg, shift_claims_per_dep_per_month, shift_claims_per_dep_total = agg_claims_per_dep_date(claims_shift, selected_year=None)

    # base_claims_received_per_seg, base_claims_received_per_month, base_claims_received_total = agg_claims_per_dep_date(claims_base,selected_year)
    # agg_claims_per_dep_date(claims_shift,selected_year)

    diff_claims = pd.merge(base_claims_received_per_seg.groupby(SEGMENT).agg({CLAIM_COUNT:"sum"}).reset_index()
                ,shift_claims_received_per_seg.groupby(SEGMENT).agg({CLAIM_COUNT:"sum"}).reset_index()
                ,on=[SEGMENT]
                ,suffixes=["_baseline","_current"])
    diff_claims['claim_diff'] = diff_claims[f"{CLAIM_COUNT}_current"] - diff_claims[f"{CLAIM_COUNT}_baseline"]
    diff_claims = diff_claims.sort_values('claim_diff',ascending=False)

    return {
        "yoy_summary_df": yoy_summary_df,
        "base_merged": base_merged,
        "shift_merged": shift_merged,
        "aoc_merged": aoc_merged,
        
        "base_claims_per_dep_per_seg":base_claims_per_dep_per_seg,
        "base_claims_per_dep_per_month":base_claims_per_dep_per_month,
        "base_claims_per_dep_total":base_claims_per_dep_total,
        
        "shift_claims_per_dep_per_seg":shift_claims_per_dep_per_seg,
        "shift_claims_per_dep_per_month":shift_claims_per_dep_per_month,
        "shift_claims_per_dep_total":shift_claims_per_dep_total,

        "aoc_claims_per_dep_per_seg":aoc_claims_per_dep_per_seg,
        "aoc_claims_per_dep_per_month":aoc_claims_per_dep_per_month,
        "aoc_claims_per_dep_total":aoc_claims_per_dep_total,
        
        "base_claims_received_per_seg": base_claims_received_per_seg,
        "base_claims_received_per_month": base_claims_received_per_month,
        "base_claims_received_total": base_claims_received_total,

        "shift_claims_received_per_seg": shift_claims_received_per_seg,
        "shift_claims_received_per_month": shift_claims_received_per_month,
        "shift_claims_received_total": shift_claims_received_total,

        "aoc_claims_received_per_seg": aoc_claims_received_per_seg,
        "aoc_claims_received_per_month": aoc_claims_received_per_month,
        "aoc_claims_received_total": aoc_claims_received_total,

        "diff_claims": diff_claims,
        "segments": segments
    }