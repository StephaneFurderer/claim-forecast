import plotly.express as px
import pandas as pd
from importlib import reload
from .constants import *
import plotly.graph_objects as go
import plotly.colors

def plot_policy_count(df, period_col, year_col, x_label, segment):
    fig = px.line(
        df,
        x=period_col,
        y=POLICY_COUNT,
        color=year_col,
        markers=True,
        labels={period_col: x_label, POLICY_COUNT: 'Policy Count', year_col: 'Depart Year'},
        title=f'Policy Count per {x_label} for Segment {segment}'
    )
    return fig

def plot_claim_count(df, period_col, year_col, x_label, segment):
    agg_df = (
        df.groupby([year_col, period_col])
            .agg({CLAIM_COUNT: 'sum'})
            .reset_index()
    )
    fig = px.line(
        agg_df,
        x=period_col,
        y=CLAIM_COUNT,
        color=year_col,
        markers=True,
        labels={period_col: x_label, CLAIM_COUNT: 'Claim Count', year_col: 'Depart Year'},
        title=f'Claim Count per {x_label} for Segment {segment}'
    )
    return fig

def plot_reported_frequency(df, period_col, year_col, x_label, segment):
    agg_df = (
        df.groupby([year_col, period_col])
            .agg({FREQUENCY_VAL: 'sum'})
            .reset_index()
    )
    #agg_df[FREQUENCY_VAL] = agg_df[CLAIM_COUNT] / agg_df[POLICY_COUNT]
 
    fig = px.line(
        agg_df,
        x=period_col,
        y=FREQUENCY_VAL,
        color=year_col,
        markers=True,
        labels={period_col: x_label, FREQUENCY_VAL: 'Reported Frequency', year_col: 'Depart Year'},
        title=f'Reported Frequency per {x_label} for Segment {segment}'
    )
    return fig 

def plot_frequency_development(df, x, segment, range_x=[-3, 24]):
    fig_dev = px.line(df, 
                     x=x, 
                     y=FREQUENCY_CUMUL,
                     color=df[DATE_DEPART_END_OF_MONTH].dt.strftime('%Y-%m'),
                     markers=True,
                     labels={DEVELOPMENT: 'Development Month',
                            FREQUENCY_CUMUL: 'Cumulative Frequency',
                            'color': 'Cohort'},
                     title=f'Frequency Development by Cohort for Segment {segment}')
    
    # Update x-axis to show negative months
    fig_dev.update_xaxes(range=range_x)
    return fig_dev

def plot_development_factors(df: pd.DataFrame, segment: str):
    """Plot development factors by cohort."""
    return px.line(
        df,
        x=DEVELOPMENT,
        y=DEVELOPMENT_FACTOR,
        color=df[DATE_DEPART_END_OF_MONTH].dt.strftime('%Y-%m'),
        markers=True,
        labels={
            DEVELOPMENT: 'Development Month',
            DEVELOPMENT_FACTOR: 'Development Factor',
            'color': 'Cohort'
        },
        title=f'Development Factors by Cohort for Segment {segment}'
    )

def plot_development_metrics(metrics: dict, title: str):
    """Plot development metrics (raw and volume-weighted averages)."""
    dfs_to_plot = []
    
    if not metrics['raw_avg'].empty:
        df_raw = metrics['raw_avg'].copy()
        df_raw['Method'] = 'Raw Average'
        dfs_to_plot.append(df_raw)
    
    if not metrics['vol_weighted_avg'].empty:
        df_vol = metrics['vol_weighted_avg'].copy()
        df_vol['Method'] = 'Volume-Weighted Average'
        dfs_to_plot.append(df_vol)
    
    if not dfs_to_plot:
        return None
        
    combined_df = pd.concat(dfs_to_plot, ignore_index=True)
    return px.line(
        combined_df,
        x=DEVELOPMENT,
        y='y',
        color='Method',
        markers=True,
        title=title,
        labels={'y': 'Development Factor', DEVELOPMENT: 'Development Period'}
    )

def plot_cumulative_metrics(metrics: dict, title: str):
    """Plot cumulative development metrics."""
    dfs_to_plot = []
    
    if not metrics['raw_cumulative'].empty:
        df_raw = metrics['raw_cumulative'].copy()
        df_raw['Method'] = 'Raw Average'
        dfs_to_plot.append(df_raw)
    
    if not metrics['vol_weighted_cumulative'].empty:
        df_vol = metrics['vol_weighted_cumulative'].copy()
        df_vol['Method'] = 'Volume-Weighted Average'
        dfs_to_plot.append(df_vol)
    
    if not dfs_to_plot:
        return None
        
    combined_df = pd.concat(dfs_to_plot, ignore_index=True)
    return px.line(
        combined_df,
        x=DEVELOPMENT,
        y='y',
        color='Method',
        markers=True,
        title=title,
        labels={'y': 'Cumulative Development Factor', DEVELOPMENT: 'Development Period'}
    )

# plot the cumulative development pattern
def plot_cumulative_development_pattern(df, x, y, segment):
    fig_dev = px.line(df, 
                     x=x, 
                     y=y,
                     markers=True,
                     labels={DEVELOPMENT: 'Development Month',
                             'y': 'Cumulative Development Factor'},
                     title=f'Cumulative Development Pattern by Cohort for Segment {segment}')
    return fig_dev

def plot_current_vs_ultimate(df: pd.DataFrame, segment: str):
    """Plot current vs ultimate frequencies by cohort for a specific segment."""
    if df.empty:
        return None

    # Filter for the specified segment.
    # Assumes SEGMENT is a column name defined as a constant (e.g., from constants.py)
    # and is available in the current scope.
    segment_df = df[df[SEGMENT] == segment].copy() 

    if segment_df.empty:
        # If no data for the specified segment, return None.
        # This handles cases where the segment exists but has no data in the input df,
        # or the segment itself is not present.
        return None
        
    # Prepare data for plotting by melting current and ultimate frequencies into a long format.
    # This creates a 'Type' column ('current_frequency', 'ultimate_frequency') and a 'Frequency' column.
    plot_df = pd.melt(
        segment_df,
        id_vars=[DATE_DEPART_END_OF_MONTH], # After filtering, SEGMENT column is constant for segment_df.
                                           # DATE_DEPART_END_OF_MONTH is used to derive 'Cohort'.
        value_vars=['current_frequency', 'ultimate_frequency'],
        var_name='Type', # This column will be used for coloring the lines.
        value_name='Frequency'
    )
    
    # Format the date as 'YYYY-MM' for a cleaner x-axis display. This becomes the 'Cohort'.
    plot_df['Cohort'] = plot_df[DATE_DEPART_END_OF_MONTH].dt.strftime('%Y-%m')
    
    # Sort by 'Cohort' to ensure lines are drawn in chronological order on the x-axis.
    # While Plotly Express px.line sorts x-axis values by default for each trace,
    # explicit sorting of the DataFrame is a good practice for clarity and robustness.
    plot_df.sort_values(by='Cohort', inplace=True) 
    
    # Create the line plot.
    # px.line is used to draw two lines: one for 'current_frequency' and one for 'ultimate_frequency'.
    fig = px.line( 
        plot_df,
        x='Cohort',
        y='Frequency',
        color='Type', # This argument maps the 'Type' column to different line colors.
        markers=True, # Adds markers to data points, consistent with other plotting functions.
        title=f'Current vs Ultimate Frequency for Segment {segment}',
        labels={
            'Frequency': 'Frequency Value', # Updated label for clarity
            'Type': 'Frequency Type',       # Updated label for clarity
            'Cohort': 'Cohort (YYYY-MM)'    # Updated label for clarity
        }
        # The 'barmode' parameter from the original call has been removed as it is specific to bar charts (px.bar)
        # and not applicable to px.line.
    )
    return fig

def plot_projected_development(df: pd.DataFrame, segment: str):
    """
    Plot observed vs projected cumulative development for all cohorts in a segment.
    Solid lines for observed, dashed for projected.
    """
    if df.empty:
        return None
    # Filter for segment
    seg_df = df[df[SEGMENT] == segment].copy()
    if seg_df.empty:
        return None
    # Prepare cohort label
    seg_df['Cohort'] = seg_df[DATE_DEPART_END_OF_MONTH].dt.strftime('%Y-%m')
    # Plot
    fig = go.Figure()
    for cohort, group in seg_df.groupby('Cohort'):
        group = group.sort_values(DEVELOPMENT)
        # Split observed and projected
        obs = group[group['is_observed']]
        proj = group[~group['is_observed']]
        # Plot observed (solid)
        if not obs.empty:
            fig.add_trace(go.Scatter(
                x=obs[DEVELOPMENT], y=obs[FREQUENCY_CUMUL],
                mode='lines+markers',
                name=f'{cohort} (observed)',
                line=dict(dash='solid'),
                legendgroup=cohort
            ))
        # Plot projected (dashed)
        if not proj.empty:
            fig.add_trace(go.Scatter(
                x=proj[DEVELOPMENT], y=proj[FREQUENCY_CUMUL],
                mode='lines+markers',
                name=f'{cohort} (projected)',
                line=dict(dash='dash'),
                legendgroup=cohort,
                showlegend=True
            ))
    fig.update_layout(
        title=f'Observed vs Projected Development for Segment {segment}',
        xaxis_title='Development Period',
        yaxis_title='Cumulative Frequency',
        legend_title='Cohort',
        template='plotly_white'
    )
    return fig

def plot_forecast_trend(
    df: pd.DataFrame,
    segment: str,
    reported_df: pd.DataFrame = None,
    exclude_date_ranges: list = None,
    min_max_development: int = 0,
    cohort_dev_df: pd.DataFrame = None
):
    """
    Plot forecast trend for a specific segment, and optionally overlay observed frequency.
    """
    import plotly.colors

    def filter_reported(input_df):
        filtered = input_df.copy()
        # Create a period column for robust month comparison
        cols = filtered.reset_index().columns
        print(cols)
        # Exclude by date ranges
        if DATE_DEPART_END_OF_MONTH in cols:
            filtered[DEPARTURE_YEAR] = filtered[DATE_DEPART_END_OF_MONTH].dt.year
            filtered[PERIOD] = filtered[DATE_DEPART_END_OF_MONTH].dt.month
        #     if exclude_date_ranges:
        #         # Convert cohort date to YYYY-MM string
        #         filtered['cohort_period'] = filtered[DATE_DEPART_END_OF_MONTH].dt.strftime('%Y-%m')
        #         exclude_months = set()
        #         for item in exclude_date_ranges:
        #             if isinstance(item, tuple):
        #                 # Range: ('2021-01', '2021-03')
        #                 start, end = pd.to_datetime(item[0]), pd.to_datetime(item[1])
        #                 months = pd.period_range(start, end, freq='M').strftime('%Y-%m').tolist()
        #                 exclude_months.update(months)
        #             else:
        #                 # Single month: '2021-08'
        #                 exclude_months.add(item)
        #         filtered = filtered[~filtered['cohort_period'].isin(exclude_months)]
        #         filtered[DEPARTURE_YEAR] = filtered[DATE_DEPART_END_OF_MONTH].dt.year
        #         filtered[PERIOD] = filtered[DATE_DEPART_END_OF_MONTH].dt.month
                
        # else:     
        filtered['cohort_period'] = pd.PeriodIndex(
            pd.to_datetime(filtered[DEPARTURE_YEAR].astype(str) + '-' + filtered[PERIOD].astype(str).str.zfill(2) + '-01'),
            freq='M'
        )
        if exclude_date_ranges:
            exclude_periods = set()
            for item in exclude_date_ranges:
                if isinstance(item, (tuple, list)):
                    start, end = pd.to_datetime(item[0]), pd.to_datetime(item[1])
                    periods = pd.period_range(start, end, freq='M')
                    exclude_periods.update(periods)
                else:
                    # Accept both 'YYYY-MM' and 'YYYY-M'
                    period = pd.Period(pd.to_datetime(item + '-01'), freq='M')
                    exclude_periods.add(period)
            filtered = filtered[~filtered['cohort_period'].isin(exclude_periods)]
        
        if cohort_dev_df is not None:
            cohort_dev_df_ = cohort_dev_df.copy()
            cohort_dev_df_['cohort_period'] = pd.PeriodIndex(
                pd.to_datetime(cohort_dev_df_[DEPARTURE_YEAR].astype(str) + '-' + cohort_dev_df_[PERIOD].astype(str).str.zfill(2) + '-01'),
                freq='M'
            )
            filtered = filtered.merge(
                cohort_dev_df_[[SEGMENT, 'cohort_period', 'max_development']],
                on=[SEGMENT, 'cohort_period'], how='left'
            )
            filtered = filtered[filtered['max_development'] >= min_max_development]
        #filtered = filtered.drop(columns=['cohort_period'])
        
        return filtered

    # Only filter the reported_df
    if reported_df is not None:
        # print(reported_df)
        reported_df = filter_reported(reported_df)
        # if DATE_DEPART_END_OF_MONTH in reported_df.columns:
        #     reported_df[DEPARTURE_YEAR] = reported_df[DATE_DEPART_END_OF_MONTH].dt.year
        #     reported_df[PERIOD] = reported_df[DATE_DEPART_END_OF_MONTH].dt.month

    if df.empty:
        return None
    seg_df = df[df[SEGMENT] == segment].copy()
    if seg_df.empty:
        return None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=seg_df[PERIOD],
        y=seg_df[ULTIMATE_FREQUENCY],
        mode='lines+markers',
        name='Ultimate Frequency (Forecast)',
        line=dict(color='blue')
    ))
    if reported_df is not None and not reported_df.empty:
        rep_seg_df = reported_df[reported_df[SEGMENT] == segment].copy()
        years = sorted(rep_seg_df[DEPARTURE_YEAR].unique())
        palette = plotly.colors.qualitative.Plotly
        for i, year in enumerate(years):
            rep_seg_df_year = rep_seg_df[rep_seg_df[DEPARTURE_YEAR] == year]
            color = palette[i % len(palette)]
            fig.add_trace(go.Scatter(
                x=rep_seg_df_year[PERIOD],
                y=rep_seg_df_year[ULTIMATE_FREQUENCY],
                mode='lines+markers',
                name=f'Reported Frequency ({year})',
                line=dict(color=color),
                legendgroup=str(year),
                showlegend=True
                
            ))
    fig.update_layout(
        title="Forecast Trend vs. Ultimate Frequency (Chain-Ladder)",
        xaxis_title="Period",
        yaxis_title="Frequency",
        legend_title="Type",
        hovermode= "x unified"
    )
    return fig

def plot_best_frequency(best_frequencies: pd.DataFrame, segment: str,y:str = 'best_frequency'):
    fig = px.line(
        best_frequencies,
        x='month',
        y=y,
        color='year',
        markers=True,
        title=f"Best Frequency Estimates for {segment}",
        labels={
            y: 'Selected Frequency',
            'month': 'Month',
            'year': 'Year'
        }
    )
    fig.update_layout(title="Best Frequency at Ultimate Selected",xaxis_title="Period",
        yaxis_title="Frequency",
        legend_title="Departure Year",hovermode= "x unified")
    # fig.update_layout(
    #     showlegend=True,
    #     legend=dict(
    #         yanchor="top",
    #         y=0.99,
    #         xanchor="left",
    #         x=0.01
    #     )
    # )
    return fig