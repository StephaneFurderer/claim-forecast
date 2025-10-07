import streamlit as st
import pandas as pd
import os
from frequency_development.constants import *
import plotly.express as px
from datetime import datetime
import json
import plotly.graph_objects as go
import numpy as np

import frequency_development as fd
from frequency_development.constants import *
from claim_count_forecast.aoc_core import analyze_aoc_data, get_segment_metrics, prepare_segment_monthly_data


ROOT_FILES = "C:\\Users\\sfurderer\\OneLake - Microsoft\\USTI-ACTUARIAL-DEV\\USTI_IDEA_SILVER.Lakehouse\\Files\\"
ROOT_POLICY_FORECAST = ROOT_FILES + "policy_count_forecast\\_results\\"
ROOT_OUTPUT_POL_FORECAST = "C:\\Users\\sfurderer\\OneLake - Microsoft\\USTI-ACTUARIAL-DEV\\USTI_IDEA_SILVER.Lakehouse\\Files\\policy_count_forecast\\" 
config_path_lag = ROOT_OUTPUT_POL_FORECAST + "config_lag.json"

scenarios = pd.read_json('scenarios.json')

# scenarios = {
#     "F1 to F2 (2025)": {
#         "baseline" : {
#             "cutoff":"2025-03-31",
#             "cutoff_finance" : "2025-03-31",
#             "cutoff_frequency":"2025-03-31",
#         },
#         "current": {
#             "cutoff":"2025-05-31",
#             "cutoff_finance" : "2025-03-31",
#             "cutoff_frequency":"2025-05-31",
#         },
#         "aoc" : {
#             "cutoff":"2025-03-31",
#             "cutoff_finance" : "2025-03-31",
#             "cutoff_frequency":"2025-05-31",
#         }
#     }
# }

st.set_page_config(layout="wide")
st.sidebar.title("Controls")


backup = st.sidebar.toggle("backup mode",value=True,disabled=True)

situation = st.sidebar.selectbox("situation under analysis",scenarios.keys(),index=0)
st.sidebar.text(f"situation: {situation}")

ROOT_FILES = "C:\\Users\\sfurderer\\OneLake - Microsoft\\USTI-ACTUARIAL-DEV\\USTI_IDEA_SILVER.Lakehouse\\Files\\"
ROOT_CLAIM_FORECAST = ROOT_FILES + "claim_count_forecast\\_results\\"
ROOT_CLAIM_FORECAST = ROOT_FILES + "claim_count_forecast\\_results\\"

# remove the segments still in heavy development to model them apart with proxy in excel
irrelevant_segments = ['Booking.com']
# Load data
#@st.cache_data
def get_policy_count_per_dep(cutoff:str, cutoff_finance:str):
    pol_dep_ = pd.read_csv(ROOT_POLICY_FORECAST + "pol_count_per_dep_.csv")
    pol_dep_["cutoff"] = pd.to_datetime(pol_dep_["cutoff"])
    pol_dep_["cutoff_finance"] = pd.to_datetime(pol_dep_["cutoff_finance"])
    pol_dep_[DATE_DEPART_END_OF_MONTH] = pd.to_datetime(pol_dep_[DATE_DEPART_END_OF_MONTH])
    pol_dep_ = pol_dep_[~(pol_dep_[SEGMENT].isin(irrelevant_segments))]
    pol_dep_ = pol_dep_[(pol_dep_["cutoff"]==cutoff) & (pol_dep_["cutoff_finance"]==cutoff_finance)]
    return pol_dep_[["cutoff","cutoff_finance",SEGMENT,DATE_DEPART_END_OF_MONTH,"idpol_unique_"]].rename(columns={'idpol_unique_': POLICY_COUNT})

#@st.cache_data
def get_claim_count_per_dep(cutoff:str, cutoff_finance:str, cutoff_frequency:str):
    claim_dep_ = pd.read_csv(ROOT_CLAIM_FORECAST + "final_claims_dep_data.csv")
    claim_dep_["cutoff"] = pd.to_datetime(claim_dep_["cutoff"])
    claim_dep_["cutoff_finance"] = pd.to_datetime(claim_dep_["cutoff_finance"])
    claim_dep_["cutoff_frequency"] = pd.to_datetime(claim_dep_["cutoff_frequency"])
    claim_dep_[DATE_DEPART_END_OF_MONTH] = pd.to_datetime(claim_dep_[DATE_DEPART_END_OF_MONTH])
    claim_dep_ = claim_dep_[~(claim_dep_[SEGMENT].isin(irrelevant_segments))]
    claim_dep_ = claim_dep_[(claim_dep_["cutoff"]==cutoff) & (claim_dep_["cutoff_finance"]==cutoff_finance) & (claim_dep_["cutoff_frequency"]==cutoff_frequency)]
    
    return claim_dep_[["cutoff","cutoff_finance","cutoff_frequency",SEGMENT,DATE_DEPART_END_OF_MONTH,"clmNum_unique_"]].rename(columns={'clmNum_unique_': CLAIM_COUNT})

def get_claim_count_per_received(cutoff:str, cutoff_finance:str, cutoff_frequency:str):
    claim_rec_ = pd.read_csv(ROOT_CLAIM_FORECAST + "final_claims_rec_data.csv")
    claim_rec_["cutoff"] = pd.to_datetime(claim_rec_["cutoff"])
    claim_rec_["cutoff_finance"] = pd.to_datetime(claim_rec_["cutoff_finance"])
    claim_rec_["cutoff_frequency"] = pd.to_datetime(claim_rec_["cutoff_frequency"])
    claim_rec_[DATE_RECEIVED_END_OF_MONTH] = pd.to_datetime(claim_rec_[DATE_RECEIVED_END_OF_MONTH])
    claim_rec_ = claim_rec_[~(claim_rec_[SEGMENT].isin(irrelevant_segments))]
    claim_rec_ = claim_rec_[(claim_rec_["cutoff"]==cutoff) & (claim_rec_["cutoff_finance"]==cutoff_finance) & (claim_rec_["cutoff_frequency"]==cutoff_frequency)]
    
    return claim_rec_[["cutoff","cutoff_finance","cutoff_frequency",SEGMENT,DATE_RECEIVED_END_OF_MONTH,"clmNum_unique_"]].rename(columns={'clmNum_unique_': CLAIM_COUNT})


scenario_for_situation = scenarios[situation]
baseline_scenario = scenario_for_situation["baseline"]
current_scenario = scenario_for_situation["current"]
aoc_scenario = scenario_for_situation["aoc"]

current_year = pd.to_datetime(current_scenario['cutoff']).year
end_window_year = pd.to_datetime(current_scenario['cutoff']).year + 1

if backup:
    policies_base = get_policy_count_per_dep(cutoff = pd.to_datetime(baseline_scenario["cutoff"])
                                             , cutoff_finance=pd.to_datetime(baseline_scenario["cutoff_finance"]))
    
    policies_shift = get_policy_count_per_dep(cutoff = pd.to_datetime(current_scenario["cutoff"])
                                             , cutoff_finance=pd.to_datetime(current_scenario["cutoff_finance"]))
    
    policies_aoc = get_policy_count_per_dep(cutoff = pd.to_datetime(aoc_scenario["cutoff"])
                                             , cutoff_finance=pd.to_datetime(aoc_scenario["cutoff_finance"]))
    
    claims_base = get_claim_count_per_dep(cutoff=pd.to_datetime(baseline_scenario["cutoff"])
                                          , cutoff_finance= pd.to_datetime(baseline_scenario["cutoff_finance"])
                                          , cutoff_frequency= pd.to_datetime(baseline_scenario["cutoff_frequency"]))
    
    claims_shift = get_claim_count_per_dep(cutoff=pd.to_datetime(current_scenario["cutoff"])
                                          , cutoff_finance= pd.to_datetime(current_scenario["cutoff_finance"])
                                          , cutoff_frequency= pd.to_datetime(current_scenario["cutoff_frequency"]))
    
    claims_aoc = get_claim_count_per_dep(cutoff=pd.to_datetime(aoc_scenario["cutoff"])
                                          , cutoff_finance= pd.to_datetime(aoc_scenario["cutoff_finance"])
                                          , cutoff_frequency= pd.to_datetime(aoc_scenario["cutoff_frequency"]))
    
    
    claims_base_received = get_claim_count_per_received(cutoff=pd.to_datetime(baseline_scenario["cutoff"])
                                          , cutoff_finance= pd.to_datetime(baseline_scenario["cutoff_finance"])
                                          , cutoff_frequency= pd.to_datetime(baseline_scenario["cutoff_frequency"]))
    
    claims_shift_received = get_claim_count_per_received(cutoff=pd.to_datetime(current_scenario["cutoff"])
                                          , cutoff_finance= pd.to_datetime(current_scenario["cutoff_finance"])
                                          , cutoff_frequency= pd.to_datetime(current_scenario["cutoff_frequency"]))
    
    claims_aoc_received = get_claim_count_per_received(cutoff=pd.to_datetime(aoc_scenario["cutoff"])
                                          , cutoff_finance= pd.to_datetime(aoc_scenario["cutoff_finance"])
                                          , cutoff_frequency= pd.to_datetime(aoc_scenario["cutoff_frequency"]))
    
    
    
    cutoff_date = pd.to_datetime(baseline_scenario["cutoff"])


else:
    st.warning("not implemented yet")
       

analysis_results = analyze_aoc_data(
    policies_base, policies_shift, policies_aoc,
    claims_base, claims_shift, claims_aoc,
    claims_base_received, claims_shift_received,claims_aoc_received,
    current_year, cutoff_date
)

yoy_summary_df = analysis_results["yoy_summary_df"]
base_merged = analysis_results["base_merged"]
shift_merged = analysis_results["shift_merged"]
aoc_merged = analysis_results["aoc_merged"]
base_claims_received_per_seg = analysis_results["base_claims_received_per_seg"]
base_claims_received_per_month = analysis_results["base_claims_received_per_month"]
base_claims_received_total = analysis_results["base_claims_received_total"]
shift_claims_received_per_seg = analysis_results["shift_claims_received_per_seg"]
shift_claims_received_per_month = analysis_results["shift_claims_received_per_month"]
shift_claims_received_total = analysis_results["shift_claims_received_total"]

aoc_claims_received_per_seg = analysis_results["aoc_claims_received_per_seg"]
aoc_claims_received_per_month = analysis_results["aoc_claims_received_per_month"]
aoc_claims_received_total = analysis_results["aoc_claims_received_total"]

diff_claims = analysis_results["diff_claims"]
segments = analysis_results["segments"]

base_claims_per_dep_per_seg = analysis_results["base_claims_per_dep_per_seg"]
base_claims_per_dep_per_month = analysis_results["base_claims_per_dep_per_month"]
base_claims_per_dep_total = analysis_results["base_claims_per_dep_total"]

shift_claims_per_dep_per_seg = analysis_results["shift_claims_per_dep_per_seg"]
shift_claims_per_dep_per_month = analysis_results["shift_claims_per_dep_per_month"]
shift_claims_per_dep_total = analysis_results["shift_claims_per_dep_total"]

aoc_claims_per_dep_per_seg = analysis_results["aoc_claims_per_dep_per_seg"]
aoc_claims_per_dep_per_month = analysis_results["aoc_claims_per_dep_per_month"]
aoc_claims_per_dep_total = analysis_results["aoc_claims_per_dep_total"]



st.title(f'Claim Count Forecast Analysis Of Change: {situation}')




# --- Evolution Scatter Plot: Baseline to Shifted with Arrows ---


def metrics_for_waterfall(previous_df:pd.DataFrame,current_df:pd.DataFrame,aoc_df:pd.DataFrame,year, col,cutoff_current_date,cutoff_date):
    # Check if we have multiple segments
    segments_in_data = previous_df[SEGMENT].unique()
    
    metric_names = ["Previous"
                        , "Change in Frequency (backtesting)"
                        , "Change in Volume (backtesting)"
                        , "Actuals vs Forecast"
                        , "After change in Actuals v Forecast"
                        , "Change in Frequency"
                        , "Change in Volume"
                        , "Change in Forecast"
                        , "Current"
                        , "Sanity Check Total"
                        , "Sanity Check Subtotal"
                        , "Sanity Check (backtesting)"
                        , "lag"
                        , "start of backtesting window"
                        ]
    
    #if len(segments_in_data) > 1:
        # Multiple segments - return segment-level decomposition
    segment_results = {}
    
    current_year = pd.to_datetime(cutoff_date).year
    previous_year = current_year - 1
    
    i = 0
    for segment in segments_in_data:
        # Filter data for this segment
        
        
        all_configs_lag = fd.load_user_config(config_path_lag)
        
        config_key_lag = f"{segment}__{str(cutoff_date.strftime('%Y-%m-%d'))}"
        user_config_lag = all_configs_lag.get(config_key_lag, None)


        if user_config_lag is not None:
            lag = user_config_lag.get("lag",0)
        else:
            lag = 0

        prev_seg = previous_df[previous_df[SEGMENT] == segment]
        curr_seg = current_df[current_df[SEGMENT] == segment]
        aoc_seg = aoc_df[aoc_df[SEGMENT] == segment]

        start_date_backtesting = pd.to_datetime(cutoff_date) + pd.offsets.MonthEnd(-lag)
        #end_date_backtesting = pd.to_datetime(cutoff_date) + pd.offsets.MonthEnd(12)
        # Calculate metrics for this segment

        # previous estimated claim received for a given year
        previous = prev_seg[(prev_seg[col].dt.year == previous_year)][CLAIM_COUNT].sum()
        
        # start_up_to_cutoff = prev_seg[(prev_seg[col].dt.year == year) & (prev_seg[col]<=pd.to_datetime(cutoff_current_date))][CLAIM_COUNT].sum()
        # aoc_up_to_cutoff = aoc_seg[(aoc_seg[col].dt.year == year) & (aoc_seg[col]<=pd.to_datetime(cutoff_current_date))][CLAIM_COUNT].sum()
        # current_up_to_cutoff = curr_seg[(curr_seg[col].dt.year == year) & (curr_seg[col]<=pd.to_datetime(cutoff_current_date))][CLAIM_COUNT].sum()
        start_up_to_cutoff = prev_seg[(prev_seg[col]>=start_date_backtesting) & (prev_seg[col]<=pd.to_datetime(cutoff_current_date))][CLAIM_COUNT].sum()
        aoc_up_to_cutoff = aoc_seg[(aoc_seg[col]>=start_date_backtesting) & (aoc_seg[col]<=pd.to_datetime(cutoff_current_date))][CLAIM_COUNT].sum()
        current_up_to_cutoff = curr_seg[(curr_seg[col]>=start_date_backtesting) & (curr_seg[col]<=pd.to_datetime(cutoff_current_date))][CLAIM_COUNT].sum()

        start_since_cutoff = prev_seg[(prev_seg[col]>pd.to_datetime(cutoff_current_date))][CLAIM_COUNT].sum()
        aoc_since_cutoff = aoc_seg[(aoc_seg[col]>pd.to_datetime(cutoff_current_date))][CLAIM_COUNT].sum()
        current_since_cutoff = curr_seg[(curr_seg[col]>pd.to_datetime(cutoff_current_date))][CLAIM_COUNT].sum()

        #backtesting
        change_in_frequency_backtesting = aoc_up_to_cutoff - start_up_to_cutoff
        actual_v_forecast = current_up_to_cutoff - start_up_to_cutoff
        change_in_volume_backtesting = actual_v_forecast - change_in_frequency_backtesting
        
        # forecast since cutoff
        change_in_frequency = aoc_since_cutoff - start_since_cutoff
        forecast_current_v_previous = (current_since_cutoff - start_since_cutoff)
        change_in_volume = forecast_current_v_previous - change_in_frequency
        current = curr_seg[(curr_seg[col].dt.year == current_year)][CLAIM_COUNT].sum()
        
        y = [previous
                , change_in_frequency_backtesting
                , change_in_volume_backtesting
                , actual_v_forecast
                , previous + actual_v_forecast
                , change_in_frequency
                , change_in_volume
                , forecast_current_v_previous
                , previous + actual_v_forecast + forecast_current_v_previous
                , previous + actual_v_forecast + forecast_current_v_previous - current
                , forecast_current_v_previous -(change_in_frequency+change_in_volume)
                , actual_v_forecast - (change_in_frequency_backtesting + change_in_volume_backtesting)
                , lag
                , start_date_backtesting]
        
        segment_results[segment] = y
            
    return pd.DataFrame(segment_results, index= metric_names).T.reset_index().rename(columns={"index":SEGMENT})
   

def fig_waterfall(waterfall_df:pd.DataFrame):

    metric_names = ["Previous"
                    , "Change in Frequency (backtesting)"
                    , "Change in Volume (backtesting)"
                    , "Actuals vs Forecast"
                    , "After change in Actuals v Forecast"
                    , "Change in Frequency"
                    , "Change in Volume"
                    , "Change in Forecast"
                    , "Current"
                    ]
    
    y_values = [ waterfall_df[metric_names[0]].sum()
                ,waterfall_df[metric_names[1]].sum()
                ,waterfall_df[metric_names[2]].sum()
                ,waterfall_df[metric_names[4]].sum()
                ,waterfall_df[metric_names[5]].sum()
                ,waterfall_df[metric_names[6]].sum()
                ,waterfall_df[metric_names[8]].sum()
                ]
    
    
        # Dynamic base, rounded down to nearest 50k
    previous_total = y_values[0]
    min_value = min(y_values[0],y_values[3],y_values[6])
    base = ((min_value-1) //100 + 1) * 100 #(previous_total // 50000) * 50000

    previous_wtfall_start = y_values[0] - base
    fig_waterfall = go.Figure(go.Waterfall(
        name = "20", orientation = "v",
        measure = ["absolute", "relative", "relative","total", "relative", "relative", "total"],
        x = [[ 'Step1: Actuals v Forecast','Step1: Actuals v Forecast','Step1: Actuals v Forecast','Step1: Actuals v Forecast'
              ,'Step2: Change in Forecast','Step2: Change in Forecast','Step2: Change in Forecast']

             ,["Previous Claim Count", 'Change in Frequency (backtesting)','Change in Volume (backtesting)', "Claim Count with Actuals"
               , metric_names[5], metric_names[6], metric_names[7]]],
        textposition = "outside",
        text = [f"{y_values[0]:,.0f}"
                , f"{y_values[1]:,.0f}"
                , f"{y_values[2]:,.0f}"
                , f"{y_values[3]:,.0f}"
                , f"{y_values[4]:,.0f}"
                , f"{y_values[5]:,.0f}"
                , f"{y_values[6]:,.0f}"],
        y = [previous_wtfall_start
            , y_values[1]
            , y_values[2]
            , None
            , y_values[4]
            , y_values[5]
            , None],
        base = base,
        connector = {"line":{"color":"rgb(63, 63, 63)"}},
    ))

    fig_waterfall.update_layout(
            title = f"Claim Count Received",
            showlegend = True
    )
    
        
    return fig_waterfall
   


waterfall_df = metrics_for_waterfall(claims_base_received
                               ,claims_shift_received
                               ,claims_aoc_received
                               ,year = current_year
                               ,col = DATE_RECEIVED_END_OF_MONTH
                               ,cutoff_current_date=pd.to_datetime(current_scenario["cutoff"])
                               ,cutoff_date = pd.to_datetime(baseline_scenario["cutoff"]))

max_lag = waterfall_df['lag'].max()

start_window = waterfall_df['start of backtesting window'].min()
start_window_year = start_window.year


# Filter and sort 2025 data
yoy_summary_df_selected_year = yoy_summary_df[yoy_summary_df['Year'] == current_year][[
        SEGMENT, 
        'Policy Volume current (current)', 
        'YoY prev (baseline)', 
        'YoY current (current)', 
        'Difference Policy Volume YoY',
        'Claim Volume prev',
        'Claim Volume current',
        'Difference Claim Volume',
        'Difference Claim Volume Observed',
        'Difference Claim Volume Forecast',
        'Difference Claim Volume Total',
        'Frequency Rate (baseline)',
        'Frequency Rate (current)',
        'Frequency Rate Diff'
        ]].sort_values(by='Policy Volume current (current)')


header_col1, header_col2 = st.columns([1,1])

with header_col1:

    st.subheader("Claim Volume")
    ht1, ht2, ht3 = st.tabs(["Per Received Month","Per Departure Month",f"Yearly data ({current_year})"])
    
    
    
    with ht1: 
        def _sum_col_current_year(df,col,metric, current_year):
            return df[df[col].dt.year == current_year][metric].sum()
        
        abs_change = _sum_col_current_year(shift_claims_received_per_month,DATE_RECEIVED_END_OF_MONTH,CLAIM_COUNT,current_year) - _sum_col_current_year(base_claims_received_per_month,DATE_RECEIVED_END_OF_MONTH,CLAIM_COUNT,current_year)
        st.subheader(f"Analysis of Change for {current_year}")
        st.metric(f"Total Received Claim Count in {current_year}", f"{_sum_col_current_year(shift_claims_received_per_month,DATE_RECEIVED_END_OF_MONTH,CLAIM_COUNT,current_year):,.0f}",delta=f"{abs_change:,.0f}")

        st.write("Claim Received: backtesting window (grey zone) and forecast")
        fig = go.Figure()
        # get the maximum lag:
        

        def _filter_df(df):
            return df[(df[DATE_RECEIVED_END_OF_MONTH].dt.year >= start_window_year) & (df[DATE_RECEIVED_END_OF_MONTH].dt.year <= end_window_year)]
        
        _base_rec = _filter_df(base_claims_received_per_month)
        _shift_rec = _filter_df(shift_claims_received_per_month)
        _aoc_rec = _filter_df(aoc_claims_received_per_month)
        # Baseline points
        fig.add_trace(go.Scatter(
            x=_base_rec[DATE_RECEIVED_END_OF_MONTH],
            y=_base_rec[CLAIM_COUNT],
            mode='lines+text',
            marker=dict(color='blue'),
            name='Baseline',
            #line_shape = 'spline'
            ))

        
        fig.add_trace(go.Scatter(
            x=_aoc_rec[DATE_RECEIVED_END_OF_MONTH],
            y=_aoc_rec[CLAIM_COUNT],
            mode='lines+text',
            marker=dict(color='red'),
            name='Analysis Of Change (Change in Freq)',
            #line_shape = 'spline'
            ))
        
        # Shifted points
        fig.add_trace(go.Scatter(
            x=_shift_rec[DATE_RECEIVED_END_OF_MONTH],
            y=_shift_rec[CLAIM_COUNT],
            mode='lines+text',
            marker=dict(color='lightblue'),
            name='Current',
            #line_shape = 'spline'
            ))
        
        fig.add_vline(x=start_window.replace(day=1)
                      , line_dash = 'dash'
                      , line_color = 'darkred'
                      , line_width = 1
                      )
        
        fig.add_vline(x=pd.to_datetime(current_scenario['cutoff']).replace(day=1)
                      , line_dash = 'dash'
                      , line_color = 'darkred'
                      , line_width = 1)
        
        fig.add_vrect(x0=start_window.replace(day=1)
                      , x1= pd.to_datetime(current_scenario['cutoff']).replace(day=1)
                      , fillcolor = 'grey'
                      , opacity = 0.1
                      , layer = 'below' \
                      , line_width = 0
                      
                      )
        st.plotly_chart(fig, key = 'claim_count_received' ,use_container_width=True)
    with ht2:
        abs_change_dep = shift_claims_per_dep_total - base_claims_per_dep_total 
        st.subheader(f"Analysis of Change for {current_year}")
        st.metric(f"Total Claim Count At Departure {current_year}", f"{shift_claims_per_dep_total:,.0f}",delta=f"{abs_change_dep:,.0f}")

        st.write("Claim Received at Depature: backtesting window (grey zone) and forecast")

        def _filter_df(df):
            return df[(df[DATE_DEPART_END_OF_MONTH].dt.year >= start_window_year) & (df[DATE_DEPART_END_OF_MONTH].dt.year <= end_window_year)]
        
        _base_rec = _filter_df(base_claims_per_dep_per_month)
        _shift_rec = _filter_df(shift_claims_per_dep_per_month)
        _aoc_rec = _filter_df(aoc_claims_per_dep_per_month)

        fig = go.Figure()
        # Baseline points
        fig.add_trace(go.Scatter(
            x=_base_rec[DATE_DEPART_END_OF_MONTH],
            y=_base_rec[CLAIM_COUNT],
            mode='lines+text',
            marker=dict(color='blue'),
            name='Baseline',
            #line_shape = 'spline'
            ))

        # Shifted points
        fig.add_trace(go.Scatter(
            x=_aoc_rec[DATE_DEPART_END_OF_MONTH],
            y=_aoc_rec[CLAIM_COUNT],
            mode='lines+text',
            marker=dict(color='red'),
            name='Analysis of Change (Change in Freq)',
            #line_shape = 'spline'
            ))
        
        # Shifted points
        fig.add_trace(go.Scatter(
            x=_shift_rec[DATE_DEPART_END_OF_MONTH],
            y=_shift_rec[CLAIM_COUNT],
            mode='lines+text',
            marker=dict(color='lightblue'),
            name='Current',
            #line_shape = 'spline'
            ))
        
        fig.add_vline(x=start_window.replace(day=1)
                      , line_dash = 'dash'
                      , line_color = 'darkred'
                      , line_width = 1
                      )
        
        fig.add_vline(x=pd.to_datetime(current_scenario['cutoff']).replace(day=1)
                      , line_dash = 'dash'
                      , line_color = 'darkred'
                      , line_width = 1)
        
        fig.add_vrect(x0=start_window.replace(day=1)
                      , x1= pd.to_datetime(current_scenario['cutoff']).replace(day=1)
                      , fillcolor = 'grey'
                      , opacity = 0.1
                      , layer = 'below' \
                      , line_width = 0
                      
                      )
        
        st.plotly_chart(fig, key = 'claim_count_departed' ,use_container_width=True)
    with ht3:
        st.dataframe(diff_claims)


    

    
    #st.write(f"Previous claim count {base_claims_received_total:.0f}")
    
    # st.subheader("Difference by Segment")
    # polar_col1 = st.columns(1)

    # # Data for plots
    # r_pos = [val if val>0 else 0 for val in diff_claims['claim_diff']]
    # r_neg = [abs(val) if val<=0 else 0 for val in diff_claims['claim_diff']]
    # theta = diff_claims[SEGMENT].tolist()

    # # pos_diffs = diff_claims[diff_claims['claim_diff'] > 0]
    # # neg_diffs = diff_claims[diff_claims['claim_diff'] < 0]
    # # max_diff = diff_claims['claim_diff'].abs().max()

    # fig_scatter_polar = go.Figure()

    # fig_scatter_polar.add_trace(go.Scatterpolar(
    #     r=r_pos,
    #     theta=theta,
    #     mode='markers+lines',
    #     line=dict(color='red', width=2),
    #     name='Increase in Claim Received'
    # ))

    # fig_scatter_polar.add_trace(go.Scatterpolar(
    # r=r_neg,
    # theta=theta,
    # mode='markers+lines',
    # line=dict(color='green', width=2),
    # name='Decrease in Claim Received'
    # ))

    # st.plotly_chart(fig_scatter_polar, use_container_width=False)
    
with header_col2:
    

    st.subheader("Step by Step Analysis of Change")
    ht21, ht22 = st.tabs(["Waterfall","Data"])
    with ht21:
        fig_waterfall_total = fig_waterfall(waterfall_df)
        st.plotly_chart(fig_waterfall_total, key = 'fig_waterfall_total' ,use_container_width=True)
    with ht22:
        st.dataframe(waterfall_df)




st.subheader("YoY Comparison by Segment and Year")
st.dataframe(yoy_summary_df)
st.dataframe(yoy_summary_df_selected_year)



fig = go.Figure()
# Baseline points
fig.add_trace(go.Scatter(
    x=base_merged['frequency'],
    y=base_merged['policy_volume'],
    mode='markers+text',
    marker=dict(size=12, color='blue'),
    text=base_merged[SEGMENT],
    textposition='top right',
    name='Baseline'
    ))

# Shifted points
fig.add_trace(go.Scatter(
    x=shift_merged['frequency'],
    y=shift_merged['policy_volume'],
    mode='markers+text',
    marker=dict(size=12, color='red'),
    text=shift_merged[SEGMENT],
    textposition='bottom left',
    name='Current'
    ))

# AOC points
fig.add_trace(go.Scatter(
    x=aoc_merged['frequency'],
    y=aoc_merged['policy_volume'],
    mode='markers+text',
    marker=dict(size=12, color='green', symbol='x'),
    text=aoc_merged[SEGMENT],
    textposition='top left',
    name='AOC'
    ))

fig.update_layout(
    title=f'Baseline vs Current vs AOC: Frequency vs Policy Volume ({current_year})',
    xaxis_title='Frequency (Claims/Policy)',
    yaxis_title='Policy Volume',
    legend_title='Scenario'
)

st.plotly_chart(fig, use_container_width=True)



# Segment-level detailed analysis
st.subheader("Segment-Level Detailed Analysis")
seg = st.selectbox("Select segment", segments, index=0)

with st.container():
    st.subheader(f"Segment: {seg}")
    col1, col2, col3, col4 = st.columns([1, 3, 3, 3])

    with col1:
        # Get segment metrics
        seg_metrics = get_segment_metrics(yoy_summary_df_selected_year, seg)
        if seg_metrics:
            st.metric("YoY 2025 Baseline", f"{seg_metrics['yoy_baseline']:.2f}%")
            st.metric("YoY 2025 Current", f"{seg_metrics['yoy_current']:.2f}%")
            st.metric("Difference Policy Volume YoY", f"{seg_metrics['diff_policy_volume_yoy']:.2f}%")

    with col2:
        # Policy volume chart
        base_monthly = prepare_segment_monthly_data(policies_base, seg, DATE_DEPART_END_OF_MONTH, POLICY_COUNT, 'baseline', current_year)
        shift_monthly = prepare_segment_monthly_data(policies_shift, seg, DATE_DEPART_END_OF_MONTH, POLICY_COUNT, 'current', current_year)
        aoc_monthly = prepare_segment_monthly_data(policies_aoc, seg, DATE_DEPART_END_OF_MONTH, POLICY_COUNT, 'aoc', current_year)
        
        merged_policies = pd.concat([base_monthly, shift_monthly, aoc_monthly])
        
        fig = px.line(merged_policies, x='month', y=POLICY_COUNT, color='scenario_year', 
                     title=f'Policy Volume for {seg} ({current_year})')
        st.plotly_chart(fig, use_container_width=True)

       

    with col3:
        # Claim count chart
        base_claims_monthly = prepare_segment_monthly_data(claims_base, seg, DATE_DEPART_END_OF_MONTH, CLAIM_COUNT, 'baseline', current_year)
        shift_claims_monthly = prepare_segment_monthly_data(claims_shift, seg, DATE_DEPART_END_OF_MONTH, CLAIM_COUNT, 'current', current_year)
        aoc_claims_monthly = prepare_segment_monthly_data(claims_aoc, seg, DATE_DEPART_END_OF_MONTH, CLAIM_COUNT, 'aoc', current_year)
        
        merged_claims = pd.concat([base_claims_monthly, shift_claims_monthly, aoc_claims_monthly])
        
        fig = px.line(merged_claims, x='month', y=CLAIM_COUNT, color='scenario_year', 
                     title=f'Claim Count for {seg} ({current_year})')
        st.plotly_chart(fig, use_container_width=True)




    with col4:
        # Frequency rate chart
        # Merge policies and claims data for frequency calculation
        merged_freq = pd.merge(
            merged_policies,
            merged_claims,
            on=['month', 'scenario_year', 'year'],
            suffixes=('_policies', '_claims')
        )
        # Calculate frequency rate
        merged_freq['frequency_rate'] = merged_freq[CLAIM_COUNT] / merged_freq[POLICY_COUNT]

        fig = px.line(merged_freq, x='month', y='frequency_rate', color='scenario_year',
                     title=f'Frequency Rate for {seg} ({current_year})')
        fig.update_layout(yaxis_title='Frequency Rate (Claims/Policy)')
        st.plotly_chart(fig, use_container_width=True)


with st.container():
    col1, col2, col3, col4 = st.columns([1, 3, 3, 3])
    with col2:
        fig_waterfall_seg = fig_waterfall(waterfall_df[waterfall_df[SEGMENT]==seg])
        st.plotly_chart(fig_waterfall_seg, use_container_width=True)
    with col3:  
        base_claims_monthly = prepare_segment_monthly_data(claims_base_received, seg, DATE_RECEIVED_END_OF_MONTH, CLAIM_COUNT, 'baseline', current_year)
        shift_claims_monthly = prepare_segment_monthly_data(claims_shift_received, seg, DATE_RECEIVED_END_OF_MONTH, CLAIM_COUNT, 'current', current_year)
        aoc_claims_monthly = prepare_segment_monthly_data(claims_aoc_received, seg, DATE_RECEIVED_END_OF_MONTH, CLAIM_COUNT, 'aoc', current_year)
        
        merged_claims = pd.concat([base_claims_monthly, shift_claims_monthly, aoc_claims_monthly])
        
        fig = px.line(merged_claims, x='month', y=CLAIM_COUNT, color='scenario_year', 
                     title=f'Claim Count Received for {seg} ({current_year})')
        st.plotly_chart(fig, use_container_width=True)