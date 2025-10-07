import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import numpy as np
import json
import os
import plotly.graph_objects as go
import shutil

import frequency_development as fd
from frequency_development.constants import *

from policy_count_forecast.core import (
    get_training_cohorts,main_pol_cohorts_development,calculate_month_difference,_forecast_pol,forecast_policy_count,load_data,get_gcp_per_pol_from_finance,get_or_create_config_for_key)




st.set_page_config(layout="wide")

#ROOT = "C:\\Users\\sfurderer\\OneLake - Microsoft\\USTI-ACTUARIAL-DEV\\USTI_IDEA_SILVER.Lakehouse\\Tables\\analysis\\"
#ROOT_OUTPUT_POL_FORECAST = "C:\\Users\\sfurderer\\OneLake - Microsoft\\USTI-ACTUARIAL-DEV\\USTI_IDEA_SILVER.Lakehouse\\Files\\policy_count_forecast\\" 
#ROOT_FILES = r"C:\Users\sfurderer\OneDrive - Generali Global Assistance\Actuarial Analytics\1 Projects\UnifiedClaimCount\backup_app\\"


ROOT_FILES = "C:\\Users\\sfurderer\\OneLake - Microsoft\\USTI-ACTUARIAL-DEV\\USTI_IDEA_SILVER.Lakehouse\\Files\\"

ROOT_OUTPUT_POL_FORECAST = ROOT_FILES + "policy_count_forecast\\"
FINANCE_INPUT_FOLDER = ROOT_OUTPUT_POL_FORECAST + "input_finance\\"
result_path = ROOT_OUTPUT_POL_FORECAST + "_results\\"
config_path = ROOT_OUTPUT_POL_FORECAST + "config_lag.json"

ROOT_BACKUP_MODE = ROOT_FILES + "_data\\"
INPUT_BACKUP_MODE_CSA = ROOT_BACKUP_MODE + "csa\\"
INPUT_BACKUP_MODE_TM = ROOT_BACKUP_MODE + "tripmate\\"

ROOT_OUTPUT_FREQUENCY = ROOT_FILES + "frequency_forecast\\"

#ROOT_OUTPUT_FREQUENCY = "C:\\Users\\sfurderer\\OneLake - Microsoft\\USTI-ACTUARIAL-DEV\\USTI_IDEA_SILVER.Lakehouse\\Files\\frequency_forecast\\" 
# ROOT_OUTPUT_FREQUENCY = r"C:\Users\sfurderer\OneDrive - Generali Global Assistance\Actuarial Analytics\1 Projects\UnifiedClaimCount\backup_app\\frequency_forecast\\"
# ROOT_BACKUP_MODE = "C:\\Users\\sfurderer\\OneLake - Microsoft\\USTI-ACTUARIAL-DEV\\USTI_IDEA_SILVER.Lakehouse\\Files\\_data\\"

#
# @st.cache_data
def load_data_backup(cutoff_date:str):
    """
    In the case the current lakehouse is down, rollback to .csv files extracted using MISDB / legacy system.
    """
    print(f"load data:{INPUT_BACKUP_MODE_CSA}")
    return fd.load_data_backup(cutoff_date,backup_root=INPUT_BACKUP_MODE_CSA)
    

@st.cache_data
def load_data_backup_tripmate(cutoff_date:str):
    """
    In the case the current lakehouse is down, rollback to .csv files extracted using MISDB / legacy system.
    """
    return fd.load_data_backup_tripmate(cutoff_date,backup_root=INPUT_BACKUP_MODE_TM)

def safe_write_config(config_path, data):
    # Backup the current config
    if os.path.exists(config_path):
        shutil.copy2(config_path, config_path + '.bak')
    # Write to a temp file first
    tempname = config_path + '.tmp'
    with open(tempname,'w') as tf:
        json.dump(data, tf, indent=2)
    
    try:
        shutil.move(tempname,config_path)
    except Exception as e:
        print("error moving temp config: {e}")
        if os.path.exists(config_path + '.bak'):
            shutil.copy2(config_path + '.bak',config_path)
            print("Restored config from backup")

def _save_csv_no_duplicates(filepath, new_df, key_columns):
    """
    Append new_df to filepath, but remove any existing rows with the same key_columns as in new_df.
    """
    if os.path.exists(filepath):
        existing_df = pd.read_csv(filepath)
        existing_df['cutoff'] = pd.to_datetime(existing_df['cutoff'])
        existing_df['cutoff_finance'] = pd.to_datetime(existing_df['cutoff_finance'])
        # Remove rows that match any (segment, cutoff, cutoff_finance) in new_df
        mask = ~existing_df.set_index(key_columns).index.isin(new_df.set_index(key_columns).index)
        combined_df = pd.concat([existing_df[mask], new_df], ignore_index=True)
    else:
        combined_df = new_df
    combined_df.to_csv(filepath, index=False)


def save_data(segment, cutoff_date, cutoff_date_finance, past_future_pol_cohorts, per_app_date_, results_path="_results"):
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    r1_df = past_future_pol_cohorts.copy()
    r2_df = per_app_date_.copy()
    for df in [r1_df, r2_df]:
        df[SEGMENT] = segment
        df['cutoff'] = pd.to_datetime(cutoff_date)
        df['cutoff_finance'] = pd.to_datetime(cutoff_date_finance)

    _save_csv_no_duplicates(
        os.path.join(results_path, 'pol_count_per_dep_.csv'),
        r1_df,
        [SEGMENT, 'cutoff', 'cutoff_finance']
    )
    _save_csv_no_duplicates(
        os.path.join(results_path, 'pol_count_per_app_.csv'),
        r2_df,
        [SEGMENT, 'cutoff', 'cutoff_finance']
    )


def save_data_tm(segment, cutoff_date, cutoff_date_finance, past_future_pol_cohorts, results_path="_results"):
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    

    r1_df = past_future_pol_cohorts.copy()
    r1_df[SEGMENT] = segment
    r1_df['cutoff'] = pd.to_datetime(cutoff_date)
    r1_df['cutoff_finance'] = pd.to_datetime(cutoff_date_finance)

    _save_csv_no_duplicates(
        os.path.join(results_path, 'pol_count_per_dep_.csv'),
        r1_df,
        [SEGMENT, 'cutoff', 'cutoff_finance']
    )
# Sidebar controls

def _compute_yoy_table(policy_df:pd.DataFrame,end_valid_date:str):
            # Calculate full-year totals
            yearly = policy_df.groupby(DEPARTURE_YEAR)[POLICY_COUNT].sum().reset_index()
            yearly['YoY_growth'] = yearly[POLICY_COUNT].pct_change().fillna(0)

            ytd = policies_seg[policies_seg[DEPARTURE_MONTH] <= pd.to_datetime(end_valid_date).month].groupby(DEPARTURE_YEAR)[POLICY_COUNT].sum().reset_index()
            ytd['YTD_growth'] = ytd[POLICY_COUNT].pct_change().fillna(0)

            growth_table = pd.merge(yearly[[DEPARTURE_YEAR, POLICY_COUNT, 'YoY_growth']], ytd[[DEPARTURE_YEAR, 'YTD_growth']], on=DEPARTURE_YEAR)
            return growth_table

st.sidebar.title("Controls")

block = st.sidebar.selectbox("CSA/TM",["CSA","TM"],index=0)


if block == "CSA":
    backup = st.sidebar.toggle("backup mode",value=True)
else:
    backup = True #true as default for Trip Mate

if block == "CSA":
    if backup:
        allowed_dates_backup = []
        # list input folders for finance data
        date_folders = [
            name for name in os.listdir(INPUT_BACKUP_MODE_CSA)
            if os.path.isdir(os.path.join(INPUT_BACKUP_MODE_CSA,name))
        ]

        for folder in date_folders:
            try:
                date = pd.to_datetime(folder, format='%Y-%m-%d',errors='raise')
                allowed_dates_backup.append(date)
            except Exception:
                pass

        allowed_dates_backup_str = [d.strftime('%Y-%m-%d') for d in sorted(allowed_dates_backup)]
        cutoff_date = st.sidebar.selectbox(
            "Select Cutoff Date (for forecast projection)",
            allowed_dates_backup_str,
            index = 0,
            disabled = False
        )
        policies_df, claims_df = load_data_backup(cutoff_date)
    else:
        
        allowed_dates =[
            pd.Timestamp('2025-04-30'),
        ]

        allowed_dates_str = [d.strftime('%Y-%m-%d') for d in allowed_dates]

        cutoff_date = st.sidebar.selectbox(
            "Select Cutoff Date (for forecast projection)",
            allowed_dates_str,
            index = 0,
            disabled = True
        )
        policies_df, claims_df = load_data(cutoff_date)
    

elif block =="TM":

    

    allowed_dates_backup = []
        # list input folders for finance data
    date_folders = [
        name for name in os.listdir(INPUT_BACKUP_MODE_TM)
        if os.path.isdir(os.path.join(INPUT_BACKUP_MODE_TM,name))
    ]
    print(date_folders)
    for folder in date_folders:
        try:
            date = pd.to_datetime(folder, format='%Y-%m-%d',errors='raise')
            allowed_dates_backup.append(date)
        except Exception:
            pass

    allowed_dates_backup_str = [d.strftime('%Y-%m-%d') for d in sorted(allowed_dates_backup)]
    cutoff_date = st.sidebar.selectbox(
        "Select Cutoff Date (for forecast projection)",
        allowed_dates_backup_str,
        index = 0,
        disabled = False
    )
    policies_df, claims_df = load_data_backup_tripmate(cutoff_date)


# list input folders for finance data
date_folders = [
    name for name in os.listdir(FINANCE_INPUT_FOLDER)
    if os.path.isdir(os.path.join(FINANCE_INPUT_FOLDER,name))
]

allowed_dates_finance = []
for folder in date_folders:
    try:
        date = pd.to_datetime(folder, format='%Y-%m-%d',errors='raise')
        allowed_dates_finance.append(date)
    except Exception:
        pass

allowed_dates_finance_str = [d.strftime('%Y-%m-%d') for d in sorted(allowed_dates_finance)]

if allowed_dates_finance_str:
    cutoff_date_finance = st.sidebar.selectbox(
        "Select Cutoff Date (for finance assumption)",
        allowed_dates_finance_str,
        index = 1,
        disabled = False
    )
else:
    st.warning("No valid date folders found in input_finance")


existing_df = pd.read_csv(os.path.join(result_path, 'pol_count_per_dep_.csv'))
existing_df['cutoff'] = pd.to_datetime(existing_df['cutoff'] )
existing_df['cutoff_finance'] = pd.to_datetime(existing_df['cutoff_finance'])
if not existing_df.empty:
    st.sidebar.write("Already saved results:")
    segment_saved_table = existing_df[(existing_df["cutoff"]==pd.to_datetime(cutoff_date)) & (existing_df["cutoff_finance"]==pd.to_datetime(cutoff_date_finance))][[SEGMENT, 'cutoff', 'cutoff_finance']].drop_duplicates().sort_values(by=SEGMENT)
    st.sidebar.dataframe(segment_saved_table)
    already_saved_segments = sorted(segment_saved_table[SEGMENT].unique())
    #st.sidebar.write(already_saved_segments)
else:
    st.sidebar.write("No results saved yet.")

show_only_non_saved_segments = st.sidebar.toggle("only show non saved segment",value=True)
irrelevant = ['Unknown','Timeshare','Expedia',"Tripmate","TripMate","Identity Theft"]

list_segments= sorted(policies_df[~policies_df[SEGMENT].isin(irrelevant)][SEGMENT].unique())
if show_only_non_saved_segments:
    list_segments = sorted(list(set(list_segments) - set(already_saved_segments)))


if list_segments == []:
    st.sidebar.warning("No more segments to work on")
else:

    segment = st.sidebar.selectbox('Select Segment', sorted(list_segments))

    policies_seg = policies_df[policies_df[SEGMENT]==segment]

    if block == "CSA":

        

        gcp_per_pol = get_gcp_per_pol_from_finance(cutoff_date_finance,finance_input_folder=FINANCE_INPUT_FOLDER)

        #should be elected to delta table: st.dataframe(gcp_per_pol)

        gcp_per_pol_seg = gcp_per_pol[gcp_per_pol[SEGMENT]==segment]
        

        past_future_pol_cohorts, per_app_date_,pol_cohorts_ = forecast_policy_count(policies_seg=policies_seg
                                                                    ,segment=segment
                                                                    ,cutoff_date=cutoff_date
                                                                    ,gcp_per_pol_seg=gcp_per_pol_seg
                                                                    ,cutoff_date_finance=cutoff_date_finance)
        yoy_c1, yoy_c2 = st.columns([1,1])
        
        with yoy_c1: 
            growth_table = _compute_yoy_table(policies_seg,end_valid_date=cutoff_date)
            st.dataframe(growth_table)
        with yoy_c2: 
            policies_seg_f = past_future_pol_cohorts.copy()
            policies_seg_f[DEPARTURE_YEAR] = policies_seg_f[DATE_DEPART_END_OF_MONTH].dt.year
            policies_seg_f[POLICY_COUNT] = policies_seg_f["idpol_unique_"]
            growth_table_f = _compute_yoy_table(policies_seg_f,end_valid_date=cutoff_date)
            st.dataframe(growth_table_f)
        st.title("Policy Count Forecasts per departure and application date given financial assumptions")
        tab0, tab1, tab3,tab4 = st.tabs(["Final Policy Count Forecasts","Financial Assumptions","Policy Count Forecasts","Full Cohorts (app x Dep)"])
        with tab0:
            st.subheader("Per Departure Date")
            st.dataframe(past_future_pol_cohorts)

            st.subheader("Per App Date")
            st.dataframe(per_app_date_)

        with tab1:
            st.subheader("forecast from finance")
            st.dataframe(gcp_per_pol_seg)

        with tab3:
            
            st.subheader("Per Departure Date")
            fig_pol_dep_date = px.line(past_future_pol_cohorts,x='dateDepart_EndOfMonth',y='idpol_unique_',color='pol_past_present')
            st.plotly_chart(fig_pol_dep_date, use_container_width=True, key="fig_pol_dep_date")
            
            st.subheader("Per Application Date")
            
            fig_pol_app_date = px.line(per_app_date_,x=DATE_SOLD_END_OF_MONTH,y='idpol_unique_',color='pol_past_present')
            st.plotly_chart(fig_pol_app_date, use_container_width=True, key="fig_pol_app_date")

            st.subheader(f"Per Application Date as of {cutoff_date} (including Finance assumptions as of {cutoff_date_finance})")
            fig_main = go.Figure()
            fig_main.add_trace(go.Scatter(
                x=per_app_date_[DATE_SOLD_END_OF_MONTH],
                y=per_app_date_['idpol_unique_'],
                mode='lines+markers',
                name=f'Model Forecast (as of {cutoff_date})',
                line=dict(color='#EF553B', dash='dash')
            ))
            fig_main.add_trace(go.Scatter(
                x=gcp_per_pol_seg[DATE_SOLD_END_OF_MONTH],
                y=gcp_per_pol_seg[POLICY_COUNT],
                mode='lines+markers',
                name=f'Finance Forecast (as of {cutoff_date_finance})',
                line=dict(color='#636EFA')
            ))
            fig_main.update_layout(
                title='Policy Count: Model Forecast vs Finance Forecast',
                xaxis_title='Sold Month',
                yaxis_title='Policy Count'
            )

            st.plotly_chart(fig_main, use_container_width=True,key="fig_main")
            
        with tab4:
            st.dataframe(pol_cohorts_)

    elif block == "TM":

        all_configs = fd.load_user_config(config_path)
        user_config = get_or_create_config_for_key(all_configs, segment, cutoff_date)
        

        def find_prior_config(all_configs, segment, cutoff_date):
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
            if not prior_dates:
                return None
            # Get the most recent prior date
            prior_dates.sort(reverse=True)
            return all_configs[prior_dates[0][1]]

        config_key = f"{segment}__{str(cutoff_date)}"
        user_config = all_configs.get(config_key, None)
        if user_config is None:
            # Try to find prior config for this segment
            prior_config = find_prior_config(all_configs, segment, cutoff_date)
            if prior_config is not None:
                user_config = prior_config
            else:
                user_config = {"lag": 0}
        
        #lag = user_config["lag"]
        #st.text(user_config)
        lag = st.sidebar.number_input(label="lag",value=user_config["lag"],min_value=0,max_value=12)
        growth_rates = user_config.get("growth_rates",{})

        policies_seg["idpol_unique"] = policies_seg[POLICY_COUNT]

        end_valid_date = pd.to_datetime(cutoff_date) + pd.offsets.MonthEnd(-lag)

        
            
        
        # Calculate full-year totals
        yearly = policies_seg.groupby(DEPARTURE_YEAR)[POLICY_COUNT].sum().reset_index()
        yearly['YoY_growth'] = yearly[POLICY_COUNT].pct_change().fillna(0)

        ytd = policies_seg[policies_seg[DEPARTURE_MONTH] <= pd.to_datetime(end_valid_date).month].groupby(DEPARTURE_YEAR)[POLICY_COUNT].sum().reset_index()
        ytd['YTD_growth'] = ytd[POLICY_COUNT].pct_change().fillna(0)

        growth_table = pd.merge(yearly[[DEPARTURE_YEAR, POLICY_COUNT, 'YoY_growth']], ytd[[DEPARTURE_YEAR, 'YTD_growth']], on=DEPARTURE_YEAR)
        
        last_year = pd.to_datetime(end_valid_date).year #policies_seg[policies_seg[DATE_DEPART_END_OF_MONTH]<=pd.to_datetime(end_valid_date)][DEPARTURE_YEAR].max()
        cutoff_year = pd.to_datetime(end_valid_date).year
        upcoming_years = [str(cutoff_year)] + [str(cutoff_year + i) for i in range(0, 3)]

        # Pre-populate with most recent YoY or YTD growth
        recent_yoy = growth_table.loc[growth_table[DEPARTURE_YEAR] == last_year, 'YoY_growth'].values[0]
        recent_ytd = growth_table.loc[growth_table[DEPARTURE_YEAR] == last_year, 'YTD_growth'].values[0] if cutoff_year in growth_table[DEPARTURE_YEAR].values else recent_yoy
        default_growth = recent_yoy if not np.isnan(recent_yoy) else recent_ytd

        # Prepare editable DataFrame
        if growth_rates == {}:
            growth_rates = {}
            for i,year in enumerate(upcoming_years):
                if i ==0:
                    #for cutoff year use YTD growth
                    growth_rates[year] = recent_ytd
                else:
                    growth_rates[year] = default_growth

        col1,col2,col3 = st.columns(3)
        
        st.text(f"end_valid_date:{end_valid_date}")
        with col1:
            
            st.subheader("Historical Policy Growth")
            st.dataframe(growth_table.style.format({'YoY_growth': '{:.2f}', 'YTD_growth': '{:.2f}'}))

        with col2:
            df_growth = pd.DataFrame(list(growth_rates.items()), columns=["Year", "YoY Growth (%)"]).sort_values("Year")
            st.subheader("Set YoY Growth for Upcoming Years")
            edited_df = st.data_editor(df_growth, num_rows="dynamic")
        with col3:
        # Use the edited values
            selected_growth_rates = dict(zip(edited_df["Year"].astype(str), edited_df["YoY Growth (%)"]))
            st.write("Selected YoY growth rates for upcoming years:", selected_growth_rates)


        all_configs[config_key]["lag"] = lag
        all_configs[config_key]["growth_rates"] = selected_growth_rates

        safe_write_config(config_path=config_path,data = all_configs)
        # with open(config_path, "w") as f:
        #     json.dump(all_configs, f, indent=2)

        
        
        df_combined,fig_distribution_selection = _forecast_pol(df = policies_seg
                    ,end_date=cutoff_date
                    ,lag=lag
                    ,growth_rates = selected_growth_rates)
        
        st.title("Policy Count Forecasts per departure given YoY growths")
        #tab0 = st.tabs(["Final Policy Count Forecasts"])
        #with tab0:
        pol_count_per_dep = df_combined.groupby(['pol_past_present',DATE_DEPART_END_OF_MONTH]).agg({"idpol_unique_":"sum"}).reset_index()
        pol_count_per_dep[SEGMENT] = segment
        pol_count_per_dep["cutoff"] = cutoff_date
        pol_count_per_dep["cutoff_finance"] = cutoff_date_finance
        pol_count_per_dep = pol_count_per_dep[["cutoff","cutoff_finance",SEGMENT,"pol_past_present",DATE_DEPART_END_OF_MONTH,"idpol_unique_"]].sort_values(DATE_DEPART_END_OF_MONTH)
        fig_pol_dep_date_tm = px.line(pol_count_per_dep,x=DATE_DEPART_END_OF_MONTH,y='idpol_unique_',color='pol_past_present')
        
        st.plotly_chart(fig_pol_dep_date_tm, use_container_width=True, key="fig_pol_dep_date_tm")
        st.dataframe(pol_count_per_dep)
        # 
        # st.plotly_chart(fig_distribution_selection, use_container_width=True, key="fig_distribution_selection")


    # Add a new mode to save all segments with their associated configurations
    save_all_mode = st.sidebar.checkbox("Save All Segments", value=False)

    if save_all_mode:
        if st.sidebar.button("Save All Results"):
            st.sidebar.write("Saving all segments with their configurations...")
            for segment in list_segments:
                st.sidebar.write(f"Processing segment: {segment}")
                if block == "CSA":
                    try:
                        gcp_per_pol_seg = gcp_per_pol[gcp_per_pol[SEGMENT]==segment]
                        policies_seg = policies_df[policies_df[SEGMENT]==segment]
                        past_future_pol_cohorts, per_app_date_,pol_cohorts_ = forecast_policy_count(policies_seg=policies_seg
                                                                    ,segment=segment
                                                                    ,cutoff_date=cutoff_date
                                                                    ,gcp_per_pol_seg=gcp_per_pol_seg
                                                                    ,cutoff_date_finance=cutoff_date_finance)
                        

                        save_data(segment, cutoff_date, cutoff_date_finance,past_future_pol_cohorts, per_app_date_, results_path=result_path)
                        st.sidebar.success(f"Results saved successfully for {segment}! 👍")
                    except Exception as e:
                        st.sidebar.error(f"Error saving results for {segment}: {e} 👎")
                else:
                    all_configs = fd.load_user_config(config_path)
                    user_config = get_or_create_config_for_key(all_configs, segment, cutoff_date)
                    
                    def find_prior_config(all_configs, segment, cutoff_date):
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
                        if not prior_dates:
                            return None
                        # Get the most recent prior date
                        prior_dates.sort(reverse=True)
                        return all_configs[prior_dates[0][1]]

                    config_key = f"{segment}__{str(cutoff_date)}"
                    user_config = all_configs.get(config_key, None)
                    if user_config is None:
                        # Try to find prior config for this segment
                        prior_config = find_prior_config(all_configs, segment, cutoff_date)
                        if prior_config is not None:
                            user_config = prior_config
                        else:
                            user_config = {"lag": 0}
                    
                    policies_seg = policies_df[policies_df[SEGMENT]==segment]
                    policies_seg["idpol_unique"] = policies_seg[POLICY_COUNT]
                    lag = user_config["lag"]
                    selected_growth_rates = user_config["growth_rates"]
                    df_combined,fig_distribution_selection = _forecast_pol(df = policies_seg
                                                                        ,end_date=cutoff_date
                                                                        ,lag=lag
                                                                        ,growth_rates = selected_growth_rates)
        
                    #st.title("Policy Count Forecasts per departure given YoY growths")
                    #tab0 = st.tabs(["Final Policy Count Forecasts"])
                    #with tab0:
                    pol_count_per_dep = df_combined.groupby(['pol_past_present',DATE_DEPART_END_OF_MONTH]).agg({"idpol_unique_":"sum"}).reset_index()
                    pol_count_per_dep[SEGMENT] = segment
                    pol_count_per_dep["cutoff"] = cutoff_date
                    pol_count_per_dep["cutoff_finance"] = cutoff_date_finance
                    pol_count_per_dep = pol_count_per_dep[["cutoff","cutoff_finance",SEGMENT,"pol_past_present",DATE_DEPART_END_OF_MONTH,"idpol_unique_"]].sort_values(DATE_DEPART_END_OF_MONTH)
                    #fig_pol_dep_date_tm = px.line(pol_count_per_dep,x=DATE_DEPART_END_OF_MONTH,y='idpol_unique_',color='pol_past_present')
                    
                    
                    # df_combined,fig_distribution_selection = _forecast_pol(df = policies_seg
                    # ,end_date=cutoff_date
                    # ,lag=user_config["lag"]
                    # ,growth_rates = user_config["growth_rates"])
                    
                    # pol_count_per_dep = df_combined.groupby(['pol_past_present',DATE_DEPART_END_OF_MONTH]).agg({"idpol_unique_":"sum"}).reset_index()
                    # pol_count_per_dep[SEGMENT] = segment
                    # pol_count_per_dep["cutoff"] = cutoff_date
                    # pol_count_per_dep["cutoff_finance"] = cutoff_date_finance
                    # pol_count_per_dep = pol_count_per_dep[["cutoff","cutoff_finance",SEGMENT,"pol_past_present",DATE_DEPART_END_OF_MONTH,"idpol_unique_"]].sort_values(DATE_DEPART_END_OF_MONTH)
                    save_data_tm(segment, cutoff_date, cutoff_date_finance,pol_count_per_dep, results_path=result_path)
                    st.sidebar.success(f"Results saved successfully for {segment}! 👍")
    else:
        # Original save button logic
        #save_user_config(config_path, all_configs)
        if st.sidebar.button(f"Save Results for {segment}"):
            if block == "CSA":
                try:
                    save_data(segment, cutoff_date, cutoff_date_finance, past_future_pol_cohorts, per_app_date_,results_path=result_path)
                    st.sidebar.success("Results saved successfully! 👍")
                except Exception as e:
                    st.sidebar.error(f"Error saving results: {e} 👎")

            else:
                try:
                    st.dataframe(pol_count_per_dep)
                    save_data_tm(segment, cutoff_date, cutoff_date_finance,pol_count_per_dep, results_path=result_path)
                    st.sidebar.success("Results saved successfully! 👍")
                except Exception as e:
                    st.sidebar.error(f"Error saving results: {e} 👎")