import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "helpers"))
from config import config

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
    
    min_date_a = policy_a['dateDepart_EndOfMonth'].min()
    max_date_a = policy_a['dateDepart_EndOfMonth'].max()
    min_date_b = policy_b['dateDepart_EndOfMonth'].min()
    max_date_b = policy_b['dateDepart_EndOfMonth'].max()
    
    min_date = min(min_date_a, min_date_b)
    max_date = max(max_date_a, max_date_b)
    
    date_range = st.sidebar.date_input(
        "Select Period",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
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
    tab1, tab2, tab3 = st.tabs(["📈 Policy Count", "🎯 Frequency", "📊 Claim Count"])
    
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
    
    # Tab 3: Claim Count Comparison
    with tab3:
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

else:
    st.info("Please run forecasts for both scenarios to enable comparison.")

