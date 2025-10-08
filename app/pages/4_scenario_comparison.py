import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import json
from pathlib import Path
import os

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "helpers"))
from config import config

import frequency_development as fd

st.set_page_config(page_title="Scenario Comparison", page_icon="🔄", layout="wide")

# Load scenarios
scenarios = pd.read_json(Path(__file__).parent.parent / 'scenarios.json')

st.title("🔄 Scenario Comparison")
st.markdown("Compare forecasting assumptions and results across different scenarios")

# Load data functions
@st.cache_data
def load_policy_data(segment, cutoff, cutoff_finance):
    """Load policy count forecast data"""
    path = config.get_policy_results_path() / 'pol_count_per_dep_.csv'
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df['cutoff'] = pd.to_datetime(df['cutoff'])
    df['cutoff_finance'] = pd.to_datetime(df['cutoff_finance'])
    df['dateDepart_EndOfMonth'] = pd.to_datetime(df['dateDepart_EndOfMonth'])
    
    filtered = df[
        (df['segment'] == segment) & 
        (df['cutoff'] == cutoff) &
        (df['cutoff_finance'] == cutoff_finance)
    ].copy()
    return filtered if not filtered.empty else None

@st.cache_data
def load_frequency_data(segment, cutoff):
    """Load frequency data"""
    path = config.get_frequency_results_path() / 'best_frequencies.csv'
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df['cutoff'] = pd.to_datetime(df['cutoff'])
    df['dateDepart_EndOfMonth'] = pd.to_datetime(df['dateDepart_EndOfMonth'])
    
    filtered = df[
        (df['segment'] == segment) & 
        (df['cutoff'] == cutoff)
    ].copy()
    return filtered if not filtered.empty else None

@st.cache_data
def load_claim_data(segment, cutoff, cutoff_finance, cutoff_frequency):
    """Load claim count forecast data"""
    path = config.get_claim_results_path() / 'final_claims_rec_data.csv'
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df['cutoff'] = pd.to_datetime(df['cutoff'])
    df['cutoff_finance'] = pd.to_datetime(df['cutoff_finance'])
    df['cutoff_frequency'] = pd.to_datetime(df['cutoff_frequency'])
    df['dateReceived_EndOfMonth'] = pd.to_datetime(df['dateReceived_EndOfMonth'])
    
    filtered = df[
        (df['segment'] == segment) & 
        (df['cutoff'] == cutoff) &
        (df['cutoff_finance'] == cutoff_finance) &
        (df['cutoff_frequency'] == cutoff_frequency)
    ].copy()
    return filtered if not filtered.empty else None

@st.cache_data
def load_raw_data_for_reported_freq(segment, cutoff):
    """Load raw policies and claims data to compute reported frequency"""
    # Construct backup paths
    backup_csa_path = str(config.BACKUP_MODE_CSA_PATH) + "\\"
    backup_tm_path = str(config.BACKUP_MODE_TM_PATH) + "\\"
    
    # Try both CSA and TM - the segment will exist in one of them
    backup_configs = [
        (backup_csa_path, fd.load_data_backup, "CSA"),
        (backup_tm_path, fd.load_data_backup_tripmate, "TM")
    ]
    
    for backup_path, loader_func, block_type in backup_configs:
        # Check if cutoff date folder exists
        date_folder = os.path.join(backup_path, cutoff)
        if not os.path.exists(date_folder):
            continue
        
        try:
            # Load data using appropriate loader
            policies_df, claims_df = loader_func(cutoff, backup_root=backup_path)
            
            # Check if segment exists in this dataset
            if segment not in policies_df['segment'].unique():
                continue
            
            # Filter to segment
            policies_seg = policies_df[policies_df['segment'] == segment].copy()
            claims_seg = claims_df[claims_df['segment'] == segment].copy()
            
            if not policies_seg.empty and not claims_seg.empty:
                return policies_seg, claims_seg
        except Exception as e:
            # If loading fails, try next backup path
            continue
    
    return None, None

@st.cache_data
def compute_reported_frequency(policies_df, claims_df, cutoff):
    """Compute reported (current observed) frequency per departure cohort"""
    if policies_df is None or claims_df is None:
        return None
    
    # Filter data to before cutoff
    cutoff_dt = pd.to_datetime(cutoff)
    policies_df = policies_df[policies_df['dateDepart_EndOfMonth'] <= cutoff_dt]
    claims_df = claims_df[claims_df['dateDepart_EndOfMonth'] <= cutoff_dt]
    
    # Group by departure month
    policies_by_dep = policies_df.groupby('dateDepart_EndOfMonth')['policy_count'].sum().reset_index()
    claims_by_dep = claims_df.groupby('dateDepart_EndOfMonth')['claim_count'].sum().reset_index()
    
    # Merge and calculate frequency
    reported_freq = pd.merge(
        policies_by_dep, 
        claims_by_dep, 
        on='dateDepart_EndOfMonth', 
        how='left'
    )
    
    reported_freq['claim_count'] = reported_freq['claim_count'].fillna(0)
    reported_freq['reported_frequency'] = reported_freq['claim_count'] / reported_freq['policy_count']
    reported_freq['reported_frequency'] = reported_freq['reported_frequency'].fillna(0)
    
    return reported_freq[['dateDepart_EndOfMonth', 'reported_frequency']]

# Get available segments
def get_available_segments():
    """Get list of segments from saved results"""
    segments = set()
    
    # Check policy results
    policy_path = config.get_policy_results_path() / 'pol_count_per_dep_.csv'
    if policy_path.exists():
        df = pd.read_csv(policy_path)
        segments.update(df['segment'].unique())
    
    return sorted(list(segments))

# Sidebar controls
st.sidebar.header("Controls")

# Segment selector
available_segments = get_available_segments()
if not available_segments:
    st.error("No forecast results found. Please run forecasts first.")
    st.stop()

segment = st.sidebar.selectbox("Select Segment", available_segments)

# Scenario selectors
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Scenario A")
    situation_a = st.selectbox("Situation A", list(scenarios.keys()), key="sit_a", index=0)
    scenario_a = st.selectbox("Scenario A", list(scenarios[situation_a].keys()), key="scen_a", index=0)
    scenario_a_config = scenarios[situation_a][scenario_a]
    
    st.caption(f"📅 Cutoff: {scenario_a_config['cutoff']}")
    st.caption(f"💰 Finance: {scenario_a_config['cutoff_finance']}")
    st.caption(f"📊 Frequency: {scenario_a_config['cutoff_frequency']}")

with col2:
    st.subheader("📈 Scenario B")
    situation_b = st.selectbox("Situation B", list(scenarios.keys()), key="sit_b", index=0)
    scenario_b = st.selectbox("Scenario B", list(scenarios[situation_b].keys()), key="scen_b", index=1 if len(scenarios[situation_b]) > 1 else 0)
    scenario_b_config = scenarios[situation_b][scenario_b]
    
    st.caption(f"📅 Cutoff: {scenario_b_config['cutoff']}")
    st.caption(f"💰 Finance: {scenario_b_config['cutoff_finance']}")
    st.caption(f"📊 Frequency: {scenario_b_config['cutoff_frequency']}")

# Load data for both scenarios
policy_a = load_policy_data(segment, scenario_a_config['cutoff'], scenario_a_config['cutoff_finance'])
policy_b = load_policy_data(segment, scenario_b_config['cutoff'], scenario_b_config['cutoff_finance'])

frequency_a = load_frequency_data(segment, scenario_a_config['cutoff_frequency'])
frequency_b = load_frequency_data(segment, scenario_b_config['cutoff_frequency'])

claim_a = load_claim_data(segment, scenario_a_config['cutoff'], scenario_a_config['cutoff_finance'], scenario_a_config['cutoff_frequency'])
claim_b = load_claim_data(segment, scenario_b_config['cutoff'], scenario_b_config['cutoff_finance'], scenario_b_config['cutoff_frequency'])

# Check if data exists
if policy_a is None or policy_b is None:
    st.warning("⚠️ Policy forecast data not found for one or both scenarios. Please run the policy forecast first.")
if frequency_a is None or frequency_b is None:
    st.warning("⚠️ Frequency data not found for one or both scenarios. Please run the frequency analysis first.")
if claim_a is None or claim_b is None:
    st.warning("⚠️ Claim forecast data not found for one or both scenarios. Please run the claim forecast first.")

# Continue only if we have at least some data
if policy_a is not None and policy_b is not None:
    
    # Date range selector
    st.sidebar.markdown("---")
    st.sidebar.subheader("Analysis Period")
    
    # Get data date ranges
    min_date_a = policy_a['dateDepart_EndOfMonth'].min()
    max_date_a = policy_a['dateDepart_EndOfMonth'].max()
    min_date_b = policy_b['dateDepart_EndOfMonth'].min()
    max_date_b = policy_b['dateDepart_EndOfMonth'].max()
    
    # Calculate absolute min/max from data
    absolute_min_date = min(min_date_a, min_date_b)
    absolute_max_date = max(max_date_a, max_date_b)
    
    # Calculate default range based on scenario dates
    # Start: 6 months before the min of scenario A's cutoff dates
    scenario_a_min_date = min(
        pd.Timestamp(scenario_a_config['cutoff']),
        pd.Timestamp(scenario_a_config['cutoff_finance']),
        pd.Timestamp(scenario_a_config['cutoff_frequency'])
    )
    default_start = scenario_a_min_date - pd.DateOffset(months=6)
    default_start = max(default_start, absolute_min_date)  # Don't go before data starts
    
    # End: 3 months after the max of scenario B's cutoff dates
    scenario_b_max_date = max(
        pd.Timestamp(scenario_b_config['cutoff']),
        pd.Timestamp(scenario_b_config['cutoff_finance']),
        pd.Timestamp(scenario_b_config['cutoff_frequency'])
    )
    default_end = scenario_b_max_date + pd.DateOffset(months=3)
    default_end = min(default_end, absolute_max_date)  # Don't go beyond data ends
    
    date_range = st.sidebar.date_input(
        "Select Period",
        value=(default_start, default_end),
        min_value=absolute_min_date,
        max_value=absolute_max_date
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)
    else:
        start_date = min_date
        end_date = max_date
    
    # View mode toggle
    st.sidebar.markdown("---")
    view_mode = st.sidebar.radio("View Mode", ["Overlay", "Side-by-Side"], index=0)
    
    # Calculate metrics for the selected period
    st.markdown("---")
    st.subheader("📊 Key Metrics")
    
    metric_col1, metric_col2 = st.columns(2)
    
    with metric_col1:
        st.markdown("**Scenario A**")
        
        # Policy count
        policy_a_period = policy_a[(policy_a['dateDepart_EndOfMonth'] >= start_date) & (policy_a['dateDepart_EndOfMonth'] <= end_date)]
        total_policies_a = policy_a_period['idpol_unique_'].sum()
        st.metric("Total Policy Count", f"{total_policies_a:,.0f}")
        
        # Frequency
        if frequency_a is not None:
            freq_a_period = frequency_a[(frequency_a['dateDepart_EndOfMonth'] >= start_date) & (frequency_a['dateDepart_EndOfMonth'] <= end_date)]
            avg_freq_a = freq_a_period['best_frequency'].mean()
            st.metric("Average Frequency", f"{avg_freq_a:.4f}")
        
        # Claims
        if claim_a is not None:
            claim_a_period = claim_a[(claim_a['dateReceived_EndOfMonth'] >= start_date) & (claim_a['dateReceived_EndOfMonth'] <= end_date)]
            total_claims_a = claim_a_period['clmNum_unique_'].sum()
            st.metric("Total Claim Count", f"{total_claims_a:,.0f}")
    
    with metric_col2:
        st.markdown("**Scenario B**")
        
        # Policy count
        policy_b_period = policy_b[(policy_b['dateDepart_EndOfMonth'] >= start_date) & (policy_b['dateDepart_EndOfMonth'] <= end_date)]
        total_policies_b = policy_b_period['idpol_unique_'].sum()
        delta_policies = ((total_policies_b - total_policies_a) / total_policies_a * 100) if total_policies_a > 0 else 0
        st.metric("Total Policy Count", f"{total_policies_b:,.0f}", f"{delta_policies:+.1f}%")
        
        # Frequency
        if frequency_a is not None and frequency_b is not None:
            freq_b_period = frequency_b[(frequency_b['dateDepart_EndOfMonth'] >= start_date) & (frequency_b['dateDepart_EndOfMonth'] <= end_date)]
            avg_freq_b = freq_b_period['best_frequency'].mean()
            delta_freq = ((avg_freq_b - avg_freq_a) / avg_freq_a * 100) if avg_freq_a > 0 else 0
            st.metric("Average Frequency", f"{avg_freq_b:.4f}", f"{delta_freq:+.1f}%")
        
        # Claims
        if claim_a is not None and claim_b is not None:
            claim_b_period = claim_b[(claim_b['dateReceived_EndOfMonth'] >= start_date) & (claim_b['dateReceived_EndOfMonth'] <= end_date)]
            total_claims_b = claim_b_period['clmNum_unique_'].sum()
            delta_claims = ((total_claims_b - total_claims_a) / total_claims_a * 100) if total_claims_a > 0 else 0
            st.metric("Total Claim Count", f"{total_claims_b:,.0f}", f"{delta_claims:+.1f}%")
    
    # Tabs for different comparisons
    st.markdown("---")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Policy Count", "🎯 Frequency (Ultimate)", "📊 Reported Frequency", "📋 Claim Count", "🔄 AOC Analysis"])
    
    # Tab 1: Policy Count Comparison
    with tab1:
        st.subheader("Policy Count Forecast Comparison")
        
        # Filter data to selected period
        policy_a_plot = policy_a[(policy_a['dateDepart_EndOfMonth'] >= start_date) & (policy_a['dateDepart_EndOfMonth'] <= end_date)]
        policy_b_plot = policy_b[(policy_b['dateDepart_EndOfMonth'] >= start_date) & (policy_b['dateDepart_EndOfMonth'] <= end_date)]
        
        if view_mode == "Overlay":
            # Overlay view
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=policy_a_plot['dateDepart_EndOfMonth'],
                y=policy_a_plot['idpol_unique_'],
                mode='lines',
                name=f'Scenario A: {scenario_a}',
                line=dict(color='#636EFA', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=policy_b_plot['dateDepart_EndOfMonth'],
                y=policy_b_plot['idpol_unique_'],
                mode='lines',
                name=f'Scenario B: {scenario_b}',
                line=dict(color='#EF553B', width=2)
            ))
            
            fig.update_layout(
                xaxis_title='Departure Date',
                yaxis_title='Policy Count',
                hovermode='x unified',
                height=500,
                showlegend=True,
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            # Side-by-side view
            col1, col2 = st.columns(2)
            
            with col1:
                fig_a = go.Figure()
                fig_a.add_trace(go.Scatter(
                    x=policy_a_plot['dateDepart_EndOfMonth'],
                    y=policy_a_plot['idpol_unique_'],
                    mode='lines',
                    name=f'Scenario A',
                    line=dict(color='#636EFA', width=2)
                ))
                fig_a.update_layout(
                    title=f'Scenario A: {scenario_a}',
                    xaxis_title='Departure Date',
                    yaxis_title='Policy Count',
                    height=400
                )
                st.plotly_chart(fig_a, use_container_width=True)
            
            with col2:
                fig_b = go.Figure()
                fig_b.add_trace(go.Scatter(
                    x=policy_b_plot['dateDepart_EndOfMonth'],
                    y=policy_b_plot['idpol_unique_'],
                    mode='lines',
                    name=f'Scenario B',
                    line=dict(color='#EF553B', width=2)
                ))
                fig_b.update_layout(
                    title=f'Scenario B: {scenario_b}',
                    xaxis_title='Departure Date',
                    yaxis_title='Policy Count',
                    height=400
                )
                st.plotly_chart(fig_b, use_container_width=True)
    
    # Tab 2: Frequency Comparison
    with tab2:
        st.subheader("Frequency Comparison")
        
        if frequency_a is not None and frequency_b is not None:
            # Filter data to selected period
            freq_a_plot = frequency_a[(frequency_a['dateDepart_EndOfMonth'] >= start_date) & (frequency_a['dateDepart_EndOfMonth'] <= end_date)]
            freq_b_plot = frequency_b[(frequency_b['dateDepart_EndOfMonth'] >= start_date) & (frequency_b['dateDepart_EndOfMonth'] <= end_date)]
            
            if view_mode == "Overlay":
                # Overlay view
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=freq_a_plot['dateDepart_EndOfMonth'],
                    y=freq_a_plot['best_frequency'],
                    mode='lines',
                    name=f'Scenario A: {scenario_a}',
                    line=dict(color='#636EFA', width=2)
                ))
                
                fig.add_trace(go.Scatter(
                    x=freq_b_plot['dateDepart_EndOfMonth'],
                    y=freq_b_plot['best_frequency'],
                    mode='lines',
                    name=f'Scenario B: {scenario_b}',
                    line=dict(color='#EF553B', width=2)
                ))
                
                fig.update_layout(
                    xaxis_title='Departure Date',
                    yaxis_title='Frequency',
                    hovermode='x unified',
                    height=500,
                    showlegend=True,
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                # Side-by-side view
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_a = go.Figure()
                    fig_a.add_trace(go.Scatter(
                        x=freq_a_plot['dateDepart_EndOfMonth'],
                        y=freq_a_plot['best_frequency'],
                        mode='lines',
                        name=f'Scenario A',
                        line=dict(color='#636EFA', width=2)
                    ))
                    fig_a.update_layout(
                        title=f'Scenario A: {scenario_a}',
                        xaxis_title='Departure Date',
                        yaxis_title='Frequency',
                        height=400
                    )
                    st.plotly_chart(fig_a, use_container_width=True)
                
                with col2:
                    fig_b = go.Figure()
                    fig_b.add_trace(go.Scatter(
                        x=freq_b_plot['dateDepart_EndOfMonth'],
                        y=freq_b_plot['best_frequency'],
                        mode='lines',
                        name=f'Scenario B',
                        line=dict(color='#EF553B', width=2)
                    ))
                    fig_b.update_layout(
                        title=f'Scenario B: {scenario_b}',
                        xaxis_title='Departure Date',
                        yaxis_title='Frequency',
                        height=400
                    )
                    st.plotly_chart(fig_b, use_container_width=True)
        else:
            st.warning("Frequency data not available for comparison.")
    
    # Tab 3: Reported Frequency Comparison
    with tab3:
        st.subheader("Reported Frequency Comparison (Current Observed)")
        st.caption("⚠️ Computing reported frequency from raw data - this may take a few seconds...")
        
        # Load raw data for both scenarios
        policies_a_raw, claims_a_raw = load_raw_data_for_reported_freq(segment, scenario_a_config['cutoff'])
        policies_b_raw, claims_b_raw = load_raw_data_for_reported_freq(segment, scenario_b_config['cutoff'])
        
        if policies_a_raw is not None and policies_b_raw is not None:
            # Compute reported frequency
            reported_freq_a = compute_reported_frequency(policies_a_raw, claims_a_raw, scenario_a_config['cutoff'])
            reported_freq_b = compute_reported_frequency(policies_b_raw, claims_b_raw, scenario_b_config['cutoff'])
            
            if reported_freq_a is not None and reported_freq_b is not None:
                # Filter to selected period
                reported_a_plot = reported_freq_a[
                    (reported_freq_a['dateDepart_EndOfMonth'] >= start_date) & 
                    (reported_freq_a['dateDepart_EndOfMonth'] <= end_date)
                ]
                reported_b_plot = reported_freq_b[
                    (reported_freq_b['dateDepart_EndOfMonth'] >= start_date) & 
                    (reported_freq_b['dateDepart_EndOfMonth'] <= end_date)
                ]
                
                if view_mode == "Overlay":
                    # Overlay view
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=reported_a_plot['dateDepart_EndOfMonth'],
                        y=reported_a_plot['reported_frequency'],
                        mode='lines',
                        name=f'Scenario A: {scenario_a}',
                        line=dict(color='#636EFA', width=2)
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=reported_b_plot['dateDepart_EndOfMonth'],
                        y=reported_b_plot['reported_frequency'],
                        mode='lines',
                        name=f'Scenario B: {scenario_b}',
                        line=dict(color='#EF553B', width=2)
                    ))
                    
                    fig.update_layout(
                        xaxis_title='Departure Date',
                        yaxis_title='Reported Frequency (Observed)',
                        hovermode='x unified',
                        height=500,
                        showlegend=True,
                        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    # Side-by-side view
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_a = go.Figure()
                        fig_a.add_trace(go.Scatter(
                            x=reported_a_plot['dateDepart_EndOfMonth'],
                            y=reported_a_plot['reported_frequency'],
                            mode='lines',
                            name=f'Scenario A',
                            line=dict(color='#636EFA', width=2)
                        ))
                        fig_a.update_layout(
                            title=f'Scenario A: {scenario_a}',
                            xaxis_title='Departure Date',
                            yaxis_title='Reported Frequency',
                            height=400
                        )
                        st.plotly_chart(fig_a, use_container_width=True)
                    
                    with col2:
                        fig_b = go.Figure()
                        fig_b.add_trace(go.Scatter(
                            x=reported_b_plot['dateDepart_EndOfMonth'],
                            y=reported_b_plot['reported_frequency'],
                            mode='lines',
                            name=f'Scenario B',
                            line=dict(color='#EF553B', width=2)
                        ))
                        fig_b.update_layout(
                            title=f'Scenario B: {scenario_b}',
                            xaxis_title='Departure Date',
                            yaxis_title='Reported Frequency',
                            height=400
                        )
                        st.plotly_chart(fig_b, use_container_width=True)
                
                # Show comparison with ultimate frequency if available
                if frequency_a is not None and frequency_b is not None:
                    st.markdown("---")
                    st.subheader("📊 Reported vs Ultimate Frequency")
                    
                    # View selector for this specific section
                    freq_view_mode = st.radio(
                        "View Mode", 
                        ["Overlay (All 4 Lines)", "Side-by-Side (Per Scenario)"], 
                        index=0,
                        horizontal=True,
                        key="freq_comparison_view"
                    )
                    
                    # Merge reported and ultimate for both scenarios
                    freq_comparison_a = pd.merge(
                        reported_freq_a,
                        frequency_a[['dateDepart_EndOfMonth', 'best_frequency']],
                        on='dateDepart_EndOfMonth',
                        how='inner'
                    )
                    freq_comparison_a = freq_comparison_a[
                        (freq_comparison_a['dateDepart_EndOfMonth'] >= start_date) &
                        (freq_comparison_a['dateDepart_EndOfMonth'] <= end_date)
                    ]
                    
                    freq_comparison_b = pd.merge(
                        reported_freq_b,
                        frequency_b[['dateDepart_EndOfMonth', 'best_frequency']],
                        on='dateDepart_EndOfMonth',
                        how='inner'
                    )
                    freq_comparison_b = freq_comparison_b[
                        (freq_comparison_b['dateDepart_EndOfMonth'] >= start_date) &
                        (freq_comparison_b['dateDepart_EndOfMonth'] <= end_date)
                    ]
                    
                    if freq_view_mode == "Overlay (All 4 Lines)":
                        # Create single chart with all 4 lines
                        fig_all = go.Figure()
                        
                        # Scenario A - Reported (solid blue)
                        fig_all.add_trace(go.Scatter(
                            x=freq_comparison_a['dateDepart_EndOfMonth'],
                            y=freq_comparison_a['reported_frequency'],
                            mode='lines',
                            name=f'A: {scenario_a} - Reported',
                            line=dict(color='#636EFA', width=2.5)
                        ))
                        
                        # Scenario A - Ultimate (dashed blue)
                        fig_all.add_trace(go.Scatter(
                            x=freq_comparison_a['dateDepart_EndOfMonth'],
                            y=freq_comparison_a['best_frequency'],
                            mode='lines',
                            name=f'A: {scenario_a} - Ultimate',
                            line=dict(color='#636EFA', width=2.5, dash='dash')
                        ))
                        
                        # Scenario B - Reported (solid red)
                        fig_all.add_trace(go.Scatter(
                            x=freq_comparison_b['dateDepart_EndOfMonth'],
                            y=freq_comparison_b['reported_frequency'],
                            mode='lines',
                            name=f'B: {scenario_b} - Reported',
                            line=dict(color='#EF553B', width=2.5)
                        ))
                        
                        # Scenario B - Ultimate (dashed red)
                        fig_all.add_trace(go.Scatter(
                            x=freq_comparison_b['dateDepart_EndOfMonth'],
                            y=freq_comparison_b['best_frequency'],
                            mode='lines',
                            name=f'B: {scenario_b} - Ultimate',
                            line=dict(color='#EF553B', width=2.5, dash='dash')
                        ))
                        
                        fig_all.update_layout(
                            xaxis_title='Departure Date',
                            yaxis_title='Frequency',
                            hovermode='x unified',
                            height=500
                        )
                        
                        st.plotly_chart(fig_all, use_container_width=True)
                    
                    else:
                        # Side-by-side view showing gap between reported and ultimate
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"**Scenario A: {scenario_a}**")
                            fig_comp_a = go.Figure()
                            fig_comp_a.add_trace(go.Scatter(
                                x=freq_comparison_a['dateDepart_EndOfMonth'],
                                y=freq_comparison_a['reported_frequency'],
                                mode='lines',
                                name='Reported (Observed)',
                                line=dict(color='#636EFA', width=2)
                            ))
                            fig_comp_a.add_trace(go.Scatter(
                                x=freq_comparison_a['dateDepart_EndOfMonth'],
                                y=freq_comparison_a['best_frequency'],
                                mode='lines',
                                name='Ultimate (Target)',
                                line=dict(color='#00CC96', width=2, dash='dash')
                            ))
                            fig_comp_a.update_layout(
                                xaxis_title='Departure Date',
                                yaxis_title='Frequency',
                                height=400
                            )
                            st.plotly_chart(fig_comp_a, use_container_width=True)
                        
                        with col2:
                            st.markdown(f"**Scenario B: {scenario_b}**")
                            fig_comp_b = go.Figure()
                            fig_comp_b.add_trace(go.Scatter(
                                x=freq_comparison_b['dateDepart_EndOfMonth'],
                                y=freq_comparison_b['reported_frequency'],
                                mode='lines',
                                name='Reported (Observed)',
                                line=dict(color='#EF553B', width=2)
                            ))
                            fig_comp_b.add_trace(go.Scatter(
                                x=freq_comparison_b['dateDepart_EndOfMonth'],
                                y=freq_comparison_b['best_frequency'],
                                mode='lines',
                                name='Ultimate (Target)',
                                line=dict(color='#00CC96', width=2, dash='dash')
                            ))
                            fig_comp_b.update_layout(
                                xaxis_title='Departure Date',
                                yaxis_title='Frequency',
                                height=400
                            )
                            st.plotly_chart(fig_comp_b, use_container_width=True)
            else:
                st.error("Could not compute reported frequency. Please check data availability.")
        else:
            st.warning("⚠️ Raw data not available for reported frequency calculation. Make sure backup data exists for the selected cutoff dates.")
    
    # Tab 4: Claim Count Comparison
    with tab4:
        st.subheader("Claim Count Forecast Comparison")
        
        if claim_a is not None and claim_b is not None:
            # Filter data to selected period
            claim_a_plot = claim_a[(claim_a['dateReceived_EndOfMonth'] >= start_date) & (claim_a['dateReceived_EndOfMonth'] <= end_date)]
            claim_b_plot = claim_b[(claim_b['dateReceived_EndOfMonth'] >= start_date) & (claim_b['dateReceived_EndOfMonth'] <= end_date)]
            
            if view_mode == "Overlay":
                # Overlay view
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=claim_a_plot['dateReceived_EndOfMonth'],
                    y=claim_a_plot['clmNum_unique_'],
                    mode='lines',
                    name=f'Scenario A: {scenario_a}',
                    line=dict(color='#636EFA', width=2)
                ))
                
                fig.add_trace(go.Scatter(
                    x=claim_b_plot['dateReceived_EndOfMonth'],
                    y=claim_b_plot['clmNum_unique_'],
                    mode='lines',
                    name=f'Scenario B: {scenario_b}',
                    line=dict(color='#EF553B', width=2)
                ))
                
                fig.update_layout(
                    xaxis_title='Received Date',
                    yaxis_title='Claim Count',
                    hovermode='x unified',
                    height=500,
                    showlegend=True,
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                # Side-by-side view
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_a = go.Figure()
                    fig_a.add_trace(go.Scatter(
                        x=claim_a_plot['dateReceived_EndOfMonth'],
                        y=claim_a_plot['clmNum_unique_'],
                        mode='lines',
                        name=f'Scenario A',
                        line=dict(color='#636EFA', width=2)
                    ))
                    fig_a.update_layout(
                        title=f'Scenario A: {scenario_a}',
                        xaxis_title='Received Date',
                        yaxis_title='Claim Count',
                        height=400
                    )
                    st.plotly_chart(fig_a, use_container_width=True)
                
                with col2:
                    fig_b = go.Figure()
                    fig_b.add_trace(go.Scatter(
                        x=claim_b_plot['dateReceived_EndOfMonth'],
                        y=claim_b_plot['clmNum_unique_'],
                        mode='lines',
                        name=f'Scenario B',
                        line=dict(color='#EF553B', width=2)
                    ))
                    fig_b.update_layout(
                        title=f'Scenario B: {scenario_b}',
                        xaxis_title='Received Date',
                        yaxis_title='Claim Count',
                        height=400
                    )
                    st.plotly_chart(fig_b, use_container_width=True)
        else:
            st.warning("Claim count data not available for comparison.")
    
    # Tab 5: AOC Analysis (Analysis of Change)
    with tab5:
        st.subheader("🔄 Analysis of Change (AOC)")
        st.caption("Compare baseline vs current scenarios with frequency impact decomposition")
        
        # Check if AOC scenario exists in the situation
        if 'aoc' not in scenarios[situation_a]:
            st.warning("⚠️ AOC scenario not available for this situation. Please ensure 'aoc' scenario is defined in scenarios.json")
        else:
            # Load AOC scenario data
            scenario_aoc_config = scenarios[situation_a]['aoc']
            
            st.info(f"""
            **AOC Scenario Configuration:**
            - Cutoff: {scenario_aoc_config['cutoff']} (same as baseline)
            - Finance: {scenario_aoc_config['cutoff_finance']} (same as baseline)
            - Frequency: {scenario_aoc_config['cutoff_frequency']} (same as current)
            
            This isolates the frequency effect by keeping policy assumptions from baseline but using frequency from current.
            """)
            
            # Load AOC claim data
            claim_aoc = load_claim_data(
                segment, 
                scenario_aoc_config['cutoff'], 
                scenario_aoc_config['cutoff_finance'], 
                scenario_aoc_config['cutoff_frequency']
            )
            
            if claim_a is None or claim_b is None or claim_aoc is None:
                st.error("⚠️ Missing claim forecast data. Please ensure baseline, current, and aoc scenarios have been forecasted.")
            else:
                # Get cutoff dates
                cutoff_a = pd.Timestamp(scenario_a_config['cutoff'])
                cutoff_b = pd.Timestamp(scenario_b_config['cutoff'])
                
                # Filter all scenarios to selected period
                claim_a_period = claim_a[
                    (claim_a['dateReceived_EndOfMonth'] >= start_date) & 
                    (claim_a['dateReceived_EndOfMonth'] <= end_date)
                ].copy()
                
                claim_aoc_period = claim_aoc[
                    (claim_aoc['dateReceived_EndOfMonth'] >= start_date) & 
                    (claim_aoc['dateReceived_EndOfMonth'] <= end_date)
                ].copy()
                
                claim_b_period = claim_b[
                    (claim_b['dateReceived_EndOfMonth'] >= start_date) & 
                    (claim_b['dateReceived_EndOfMonth'] <= end_date)
                ].copy()
                
                # Split into two parts based on cutoff_B
                st.markdown("---")
                
                # Part 1: Actual vs Forecast (≤ cutoff_B)
                st.subheader("📊 Part 1: Actual vs Forecast (≤ " + cutoff_b.strftime('%Y-%m-%d') + ")")
                st.caption("Comparing actual observed claims (current) against what was forecasted (baseline)")
                
                baseline_actual = claim_a_period[claim_a_period['dateReceived_EndOfMonth'] <= cutoff_b]['clmNum_unique_'].sum()
                aoc_actual = claim_aoc_period[claim_aoc_period['dateReceived_EndOfMonth'] <= cutoff_b]['clmNum_unique_'].sum()
                current_actual = claim_b_period[claim_b_period['dateReceived_EndOfMonth'] <= cutoff_b]['clmNum_unique_'].sum()
                
                freq_impact_actual = aoc_actual - baseline_actual
                vol_impact_actual = current_actual - aoc_actual
                total_impact_actual = current_actual - baseline_actual
                
                # Waterfall chart for Part 1
                fig_waterfall_1 = go.Figure(go.Waterfall(
                    name="Actual vs Forecast",
                    orientation="v",
                    measure=["absolute", "relative", "relative", "total"],
                    x=["Baseline<br>Forecast", "Frequency<br>Impact", "Volume<br>Impact", "Current<br>Actual"],
                    y=[baseline_actual, freq_impact_actual, vol_impact_actual, total_impact_actual],
                    text=[f"{baseline_actual:,.0f}", f"{freq_impact_actual:+,.0f}", f"{vol_impact_actual:+,.0f}", f"{current_actual:,.0f}"],
                    textposition="outside",
                    connector={"line": {"color": "rgb(63, 63, 63)"}},
                ))
                
                fig_waterfall_1.update_layout(
                    title="Claims Variance: Baseline Forecast → Current Actual",
                    yaxis_title="Claim Count",
                    showlegend=False,
                    height=400
                )
                
                st.plotly_chart(fig_waterfall_1, use_container_width=True)
                
                # Metrics for Part 1
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Baseline Forecast", f"{baseline_actual:,.0f}")
                with col2:
                    st.metric("Frequency Impact", f"{freq_impact_actual:+,.0f}", 
                             f"{(freq_impact_actual/baseline_actual*100):+.1f}%" if baseline_actual > 0 else "N/A")
                with col3:
                    st.metric("Volume Impact", f"{vol_impact_actual:+,.0f}",
                             f"{(vol_impact_actual/baseline_actual*100):+.1f}%" if baseline_actual > 0 else "N/A")
                with col4:
                    st.metric("Current Actual", f"{current_actual:,.0f}",
                             f"{(total_impact_actual/baseline_actual*100):+.1f}%" if baseline_actual > 0 else "N/A")
                
                # Part 2: Forecast vs Forecast (> cutoff_B)
                st.markdown("---")
                st.subheader("📈 Part 2: Remaining Forecast Change (> " + cutoff_b.strftime('%Y-%m-%d') + ")")
                st.caption("Comparing how the remaining forecast has changed between baseline and current")
                
                baseline_forecast = claim_a_period[claim_a_period['dateReceived_EndOfMonth'] > cutoff_b]['clmNum_unique_'].sum()
                aoc_forecast = claim_aoc_period[claim_aoc_period['dateReceived_EndOfMonth'] > cutoff_b]['clmNum_unique_'].sum()
                current_forecast = claim_b_period[claim_b_period['dateReceived_EndOfMonth'] > cutoff_b]['clmNum_unique_'].sum()
                
                freq_impact_forecast = aoc_forecast - baseline_forecast
                vol_impact_forecast = current_forecast - aoc_forecast
                total_impact_forecast = current_forecast - baseline_forecast
                
                # Waterfall chart for Part 2
                fig_waterfall_2 = go.Figure(go.Waterfall(
                    name="Forecast Change",
                    orientation="v",
                    measure=["absolute", "relative", "relative", "total"],
                    x=["Baseline<br>Remaining", "Frequency<br>Impact", "Volume<br>Impact", "Current<br>Remaining"],
                    y=[baseline_forecast, freq_impact_forecast, vol_impact_forecast, total_impact_forecast],
                    text=[f"{baseline_forecast:,.0f}", f"{freq_impact_forecast:+,.0f}", f"{vol_impact_forecast:+,.0f}", f"{current_forecast:,.0f}"],
                    textposition="outside",
                    connector={"line": {"color": "rgb(63, 63, 63)"}},
                ))
                
                fig_waterfall_2.update_layout(
                    title="Remaining Claims Forecast Change",
                    yaxis_title="Claim Count",
                    showlegend=False,
                    height=400
                )
                
                st.plotly_chart(fig_waterfall_2, use_container_width=True)
                
                # Metrics for Part 2
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Baseline Remaining", f"{baseline_forecast:,.0f}")
                with col2:
                    st.metric("Frequency Impact", f"{freq_impact_forecast:+,.0f}",
                             f"{(freq_impact_forecast/baseline_forecast*100):+.1f}%" if baseline_forecast > 0 else "N/A")
                with col3:
                    st.metric("Volume Impact", f"{vol_impact_forecast:+,.0f}",
                             f"{(vol_impact_forecast/baseline_forecast*100):+.1f}%" if baseline_forecast > 0 else "N/A")
                with col4:
                    st.metric("Current Remaining", f"{current_forecast:,.0f}",
                             f"{(total_impact_forecast/baseline_forecast*100):+.1f}%" if baseline_forecast > 0 else "N/A")
                
                # Summary section
                st.markdown("---")
                st.subheader("📋 Total Period Summary")
                
                total_baseline = baseline_actual + baseline_forecast
                total_current = current_actual + current_forecast
                total_freq_impact = freq_impact_actual + freq_impact_forecast
                total_vol_impact = vol_impact_actual + vol_impact_forecast
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Change", f"{(total_current - total_baseline):+,.0f}",
                             f"{((total_current - total_baseline)/total_baseline*100):+.1f}%" if total_baseline > 0 else "N/A")
                with col2:
                    st.metric("Total Frequency Impact", f"{total_freq_impact:+,.0f}",
                             f"{(total_freq_impact/total_baseline*100):+.1f}%" if total_baseline > 0 else "N/A")
                with col3:
                    st.metric("Total Volume Impact", f"{total_vol_impact:+,.0f}",
                             f"{(total_vol_impact/total_baseline*100):+.1f}%" if total_baseline > 0 else "N/A")

else:
    st.info("Please run forecasts for both scenarios to enable comparison.")

