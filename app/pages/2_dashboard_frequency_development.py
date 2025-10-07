
import pandas as pd

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import numpy as np
import json
import os
import shutil
import tempfile

# import from the installed package
import frequency_development as fd
from frequency_development import constants as const
from frequency_development.core import (
    load_data,
    flag_major_cats_pandas,
    filter_and_group_by_period,
    compute_reported_frequency,
    compute_claim_development_triangle_pandas,
    compute_frequency_dev_pandas,
    compute_development_factors,
    calculate_development_metrics,
    project_cohort_development,
    estimate_ultimate_frequencies,
    compute_raw_average_ultimate_frequency,
    select_best_frequency,
    analyze_and_save_segment,
    load_user_config,
    load_data_backup,
    save_user_config,
    get_or_create_config_for_key
)
from frequency_development.plot_utils import (
    plot_reported_frequency,
    plot_policy_count,
    plot_claim_count,
    plot_frequency_development,
    plot_cumulative_metrics,
    plot_projected_development,
    plot_current_vs_ultimate,
    plot_development_factors,
    plot_development_metrics,
    plot_forecast_trend,
    plot_best_frequency
)

# Use constants from the package
SEGMENT = const.SEGMENT
POLICY_COUNT = const.POLICY_COUNT
CLAIM_COUNT = const.CLAIM_COUNT
FREQUENCY_VAL = const.FREQUENCY_VAL
DEPARTURE_DAY = const.DEPARTURE_DAY
DEPARTURE_YEAR = const.DEPARTURE_YEAR
PERIOD = const.PERIOD
DEVELOPMENT = const.DEVELOPMENT
DATE_DEPART_END_OF_MONTH = const.DATE_DEPART_END_OF_MONTH
FREQUENCY_CUMUL = const.FREQUENCY_CUMUL
DEVELOPMENT_FACTOR = const.DEVELOPMENT_FACTOR
ULTIMATE_FREQUENCY = const.ULTIMATE_FREQUENCY
CURRENT_FREQUENCY = const.CURRENT_FREQUENCY
IS_MAJOR_CAT = const.IS_MAJOR_CAT
CAT_CODE = const.CAT_CODE
SOLD_DAY = const.SOLD_DAY
DATE_SOLD_END_OF_MONTH = const.DATE_SOLD_END_OF_MONTH
RECEIVED_DAY = const.RECEIVED_DAY
DATE_RECEIVED_END_OF_MONTH = const.DATE_RECEIVED_END_OF_MONTH



st.set_page_config(layout="wide")

ROOT = "C:\\Users\\sfurderer\\OneLake - Microsoft\\USTI-ACTUARIAL-DEV\\USTI_IDEA_SILVER.Lakehouse\\Tables\\analysis\\"

#ROOT_FILES = r"C:\Users\sfurderer\OneDrive - Generali Global Assistance\Actuarial Analytics\1 Projects\UnifiedClaimCount\backup_app\\"
ROOT_FILES = "C:\\Users\\sfurderer\\OneLake - Microsoft\\USTI-ACTUARIAL-DEV\\USTI_IDEA_SILVER.Lakehouse\\Files\\"

ROOT_OUTPUT_POL_FORECAST = ROOT_FILES + "policy_count_forecast\\"
FINANCE_INPUT_FOLDER = ROOT_OUTPUT_POL_FORECAST + "input_finance\\"
#result_path = ROOT_OUTPUT_POL_FORECAST + "_results\\"

ROOT_BACKUP_MODE = ROOT_FILES + "_data\\"
INPUT_BACKUP_MODE_CSA = ROOT_BACKUP_MODE + "csa\\"
INPUT_BACKUP_MODE_TM = ROOT_BACKUP_MODE + "tripmate\\"

#ROOT_2 = r"C:\Users\sfurderer\OneDrive - Generali Global Assistance\Actuarial Analytics\1 Projects\UnifiedClaimCount\backup_app\\"
ROOT_OUTPUT = ROOT_FILES + "frequency_forecast\\" 
config_path = ROOT_OUTPUT + "config_freq.json"
result_path = ROOT_OUTPUT 

# INPUT_BACKUP_MODE_CSA = ROOT_BACKUP_MODE + "csa\\"
# INPUT_BACKUP_MODE_TM = ROOT_BACKUP_MODE + "tripmate\\"

#ROOT_OUTPUT_POL_FORECAST = ROOT_2 + "policy_count_forecast\\" #"C:\\Users\\sfurderer\\OneLake - Microsoft\\USTI-ACTUARIAL-DEV\\USTI_IDEA_SILVER.Lakehouse\\Files\\policy_count_forecast\\" 
config_path_lag = ROOT_OUTPUT_POL_FORECAST + "config_lag.json"


def load_data(cutoff_date:str):
    """
    Load data from Microsoft Fabric pipeline (via file directory)
    """
    cutoff_date_ = cutoff_date.replace("-","_")
    claims_file = ROOT + f"_clm_count_{cutoff_date_}"
    policies_file = ROOT + f"_pol_count_{cutoff_date_}"
    claims_df = pd.read_parquet(claims_file)
    policies_df = pd.read_parquet(policies_file)

    for col in [SOLD_DAY,DEPARTURE_DAY,DATE_SOLD_END_OF_MONTH,DATE_DEPART_END_OF_MONTH]:
        policies_df[col] = pd.to_datetime(policies_df[col]).dt.tz_localize(None)
    
    for col in [SOLD_DAY,DEPARTURE_DAY,RECEIVED_DAY,DATE_SOLD_END_OF_MONTH,DATE_DEPART_END_OF_MONTH,DATE_RECEIVED_END_OF_MONTH]:
        claims_df[col] = pd.to_datetime(claims_df[col]).dt.tz_localize(None)
    
    policies_df , claims_df = fd.preprocess_data(policies_df,claims_df)
    return policies_df , claims_df

@st.cache_data
def load_data_backup(cutoff_date:str):
    """
    In the case the current lakehouse is down, rollback to .csv files extracted using MISDB / legacy system.
    """
    return fd.load_data_backup(cutoff_date,backup_root=INPUT_BACKUP_MODE_CSA)
    

@st.cache_data
def load_data_backup_tripmate(cutoff_date:str):
    """
    In the case the current lakehouse is down, rollback to .csv files extracted using MISDB / legacy system.
    """
    return fd.load_data_backup_tripmate(cutoff_date,backup_root=INPUT_BACKUP_MODE_TM)


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
    

st.sidebar.title("Controls")
block = st.sidebar.selectbox("CSA/TM",["CSA","TM"],index=0)
#st.sidebar.text(block)
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




irrelevant = ['Unknown','Timeshare','Expedia',"Tripmate","Identity Theft"]

list_segments= policies_df[~policies_df[SEGMENT].isin(irrelevant)][SEGMENT].unique()
segment = st.sidebar.selectbox('Select Segment', sorted(list_segments))
granularity = 'Month' # st.sidebar.selectbox('Granularity', ['Month', 'Week', 'Day'],disabled=True)
with_large_cat_events = st.sidebar.toggle('With Large Cat Events', value=False)
# display just FYI

if block == "CSA":
    lag = 0
else:
    #load the lag config file
    all_configs_lag = fd.load_user_config(config_path_lag)
    config_key_lag = f"{segment}__{str(cutoff_date)}"
    user_config_lag = all_configs_lag.get(config_key_lag, None)
    lag = user_config_lag.get("lag",0)

lag = st.sidebar.number_input("lag",value=lag,disabled=True,help="as defined in the policy count forecast module")


all_configs = load_user_config(config_path)
user_config = get_or_create_config_for_key(all_configs, segment, cutoff_date)


num_claims_threshold = st.sidebar.number_input('Number of Claims Threshold', value=user_config.get("num_claims_threshold",100), min_value=1, max_value=1000,disabled=False)

policies_df_seg = policies_df[policies_df[SEGMENT] == segment]
claims_df_seg = claims_df[claims_df[SEGMENT]==segment]
# Date range selector
min_date = policies_df_seg[DEPARTURE_DAY].min()
max_date = claims_df_seg[DEPARTURE_DAY].max()
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# st.subheader("claims_df_seg")
# st.dataframe(claims_df_seg)

use_volume_weighted = True # st.sidebar.toggle('Use Volume Weighted Average Development Factors', value=True)

claims_df_seg = flag_major_cats_pandas(claims_df_seg,num_claims_threshold)

policies_seg, claims_seg, x_label = filter_and_group_by_period(policies_df_seg, claims_df_seg, segment, date_range, granularity, with_large_cat_events)

# st.dataframe(claims_seg.groupby([SEGMENT,IS_MAJOR_CAT,CAT_CODE]).agg({CLAIM_COUNT: 'sum'}).reset_index())

reported_freq_all_cats = compute_reported_frequency(policies_seg
                                                      , claims_seg
                                                      , groupby_cols_policy=[SEGMENT,DEPARTURE_YEAR, PERIOD]
                                                      , groupby_cols_claim=[SEGMENT,DEPARTURE_YEAR, PERIOD,CAT_CODE])

# Split the screen into two columns
col1, col2 = st.columns(2)

with col1:
    st.header("Main Analysis")
        
    tab1, tab_cat, tab2, tab3 = st.tabs(["Current View", "Major Cat Events", "Frequency Development", "Development Factors"])

    with tab1:
        st.title('Current View')
        
        fig_freq = plot_reported_frequency(reported_freq_all_cats, PERIOD, DEPARTURE_YEAR, x_label, segment)
        st.plotly_chart(fig_freq, use_container_width=True, key="freq_chart")

        reported_freq_all_cats_agg = (
            reported_freq_all_cats.groupby([SEGMENT,DEPARTURE_YEAR, PERIOD])
                .agg({FREQUENCY_VAL: 'sum'})
                .reset_index()
        )

        fig_policies = plot_policy_count(reported_freq_all_cats, PERIOD, DEPARTURE_YEAR, x_label, segment)
        st.plotly_chart(fig_policies, use_container_width=True, key="policies_chart")

        fig_claims = plot_claim_count(reported_freq_all_cats, PERIOD, DEPARTURE_YEAR, x_label, segment)
        st.plotly_chart(fig_claims, use_container_width=True, key="claims_chart")

        
    with tab_cat:
        st.title('Major Cat Events')
        st.dataframe(claims_seg.groupby([SEGMENT,IS_MAJOR_CAT,CAT_CODE]).agg({CLAIM_COUNT: 'sum'}).reset_index())
        

    with tab2:
        st.title('Development View')
        
        # create the development triangle per cohort (departure year, period)
        dev_triangle = compute_claim_development_triangle_pandas(claims_seg,groupby_cols=[SEGMENT, DATE_DEPART_END_OF_MONTH, DEVELOPMENT])
        

        dev_policies = policies_seg.groupby([SEGMENT, DATE_DEPART_END_OF_MONTH]).agg({POLICY_COUNT: 'sum'}).reset_index()
        # compute the frequency per cohort
        freq_dev_cohort = compute_frequency_dev_pandas(dev_triangle, dev_policies, groupby_cols=[SEGMENT, DATE_DEPART_END_OF_MONTH])
        #freq_dev_cohort[DEVELOPMENT].min()
        fig_dev = plot_frequency_development(freq_dev_cohort[freq_dev_cohort[DATE_DEPART_END_OF_MONTH]<=pd.to_datetime(cutoff_date)], DEVELOPMENT, segment,range_x=[-7, 24])
        st.plotly_chart(fig_dev, use_container_width=True, key="dev_chart")

        st.subheader("Development Factors")
        st.dataframe(freq_dev_cohort)

    # compute the development factors
    with tab3:
        #st.text("Development Factors")

        # Calculate development factors
        individual_dev_factors = compute_development_factors(freq_dev_cohort)
        #st.subheader("Individual Development Factors")
        #st.dataframe(individual_dev_factors)
        # Calculate all metrics in one step
        metrics = calculate_development_metrics(individual_dev_factors)

        # if use_volume_weighted:
        #     st.subheader("Volume Weighted Average Development Factors")
        #     st.dataframe(metrics['vol_weighted_avg'])
        # else:
        #     st.subheader("Raw Average Development Factors")
        #     st.dataframe(metrics['raw_avg'])

       

        # Plot cumulative development factors
        # st.subheader("Cumulative Development Factors")
        # fig_cumulative = plot_cumulative_metrics(metrics, "Cumulative Development Factors")
        # if fig_cumulative:
        #     st.plotly_chart(fig_cumulative, use_container_width=True, key="cumulative_chart")
            
        
        # Project and plot full development for all cohorts
        st.subheader("Projected Full Development for All Cohorts")
        projected_df = project_cohort_development(freq_dev_cohort[freq_dev_cohort[DATE_DEPART_END_OF_MONTH]<=pd.to_datetime(cutoff_date)], metrics, pd.to_datetime(cutoff_date),use_volume_weighted=use_volume_weighted)
        #st.dataframe(projected_df)
        if not projected_df.empty:
            fig_proj = plot_projected_development(projected_df[projected_df[DEVELOPMENT]>-7], segment)
            if fig_proj:
                st.plotly_chart(fig_proj, use_container_width=True, key="proj_chart")
            st.dataframe(projected_df)
        else:
            st.warning("No projected development available. This could be due to insufficient data or missing development factors.")


        # Estimate ultimate frequencies
        st.subheader("Ultimate Frequency Estimation")
        ultimate_freq = estimate_ultimate_frequencies(projected_df)
        ultimate_freq = ultimate_freq.merge(dev_policies[[SEGMENT,DATE_DEPART_END_OF_MONTH,POLICY_COUNT]],on=[SEGMENT,DATE_DEPART_END_OF_MONTH],how='left')
        if not ultimate_freq.empty:
            fig_ultimate = plot_current_vs_ultimate(ultimate_freq[ultimate_freq[DATE_DEPART_END_OF_MONTH]<=pd.to_datetime(cutoff_date)], segment)
            if fig_ultimate:
                st.plotly_chart(fig_ultimate, use_container_width=True, key="ultimate_chart")
            st.dataframe(ultimate_freq)
        else:
            st.warning("No ultimate frequency estimates available. This could be due to insufficient development data.")


         # Plot individual development factors
        st.subheader("Development Factors")
        fig_dev_factors = plot_development_factors(individual_dev_factors, segment)
        st.plotly_chart(fig_dev_factors, use_container_width=True, key="dev_factors_chart")

        # Plot average development factors
        st.subheader("Average Development Factors")
        fig_avg_factors = plot_development_metrics(metrics, "Average Development Factors")
        if fig_avg_factors:
            st.plotly_chart(fig_avg_factors, use_container_width=True, key="avg_factors_chart")

    

with col2:
    
    # --- Config UI for exclusions and min_max_development (per segment and cutoff date) ---
    
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
    #st.text(config_key)
    
    user_config = all_configs.get(config_key, None)
    #st.write(user_config)
    if user_config is None:
        # Try to find prior config for this segment
        prior_config = find_prior_config(all_configs, segment, cutoff_date)
        if prior_config is not None:
            user_config = prior_config
        else:
            user_config = {"exclude_dates": [], "min_max_development": 0}

    st.subheader("Forecast Exclusion Settings")
    def exclude_dates_to_str(exclude_dates):
        items = []
        for item in exclude_dates:
            if isinstance(item, (tuple, list)) and len(item) == 2:
                items.append(f"{item[0]} to {item[1]}")
            else:
                items.append(str(item))
        return ", ".join(items)

    exclude_dates = st.text_input(
        "Exclude Cohort Months or Ranges (comma separated, e.g. 2021-01,2021-03 to 2021-05,2022-08)",
        value=exclude_dates_to_str(user_config.get("exclude_dates", []))
    )
    min_max_development = st.number_input(
        "Minimum Max Development to Include Cohort", min_value=-24, max_value=24, value=user_config.get("min_max_development", 0)
    )

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

    exclude_dates_list = parse_exclude_dates(exclude_dates)

    

    #get the max 
    # get the ultimate frequency and the current month of development (max development) per cohort
    idx = projected_df[projected_df['is_observed'] == True].groupby([SEGMENT, DATE_DEPART_END_OF_MONTH])[DEVELOPMENT].idxmax()
    #idx = projected_df.groupby([SEGMENT, DATE_DEPART_END_OF_MONTH])[DEVELOPMENT].idxmax()
    result_ultimate = projected_df.loc[idx].reset_index().rename(columns={DEVELOPMENT: 'max_development'})

    idx = projected_df.groupby([SEGMENT, DATE_DEPART_END_OF_MONTH])[DEVELOPMENT].idxmax()
    result_ultimate_r = projected_df.loc[idx][[SEGMENT, DATE_DEPART_END_OF_MONTH,FREQUENCY_CUMUL]]
    result_ultimate = result_ultimate[[SEGMENT, DATE_DEPART_END_OF_MONTH,'max_development']].merge(result_ultimate_r,on=[SEGMENT, DATE_DEPART_END_OF_MONTH],how='left')
    result_ultimate[DEPARTURE_YEAR] = result_ultimate[DATE_DEPART_END_OF_MONTH].dt.year
    result_ultimate[PERIOD] = result_ultimate[DATE_DEPART_END_OF_MONTH].dt.month


   
    
    #compute the ultimate frequency
    ultimate_freq_avg_no_filter,df_ultimate_test_no_filter = compute_raw_average_ultimate_frequency(ultimate_freq,exclude_date_ranges=None,min_max_development=-24,cohort_dev_df=result_ultimate,use_volume_weighted=use_volume_weighted)
    ultimate_freq_avg,df_ultimate_test_ = compute_raw_average_ultimate_frequency(ultimate_freq,exclude_dates_list,min_max_development,cohort_dev_df=result_ultimate,use_volume_weighted=use_volume_weighted)

    #st.subheader("debug")
    #st.dataframe(df_ultimate_test_)
    #st.subheader("Forecast Trend")
    #st.dataframe(reported_freq_all_cats_agg)
    
    # two tabs:
    tab1, tab2, tab3, tab4 = st.tabs(["All Cohorts", "Filtered Cohorts", "Best Frequencies","result ultimate data"])
    with tab1:
        #st.dataframe(ultimate_freq_avg)
        fig = plot_forecast_trend(ultimate_freq_avg, segment, reported_df=df_ultimate_test_no_filter)
        st.plotly_chart(fig, use_container_width=True, key="forecast_trend_all")
    with tab2:
        fig = plot_forecast_trend(ultimate_freq_avg, segment, reported_df=df_ultimate_test_,exclude_date_ranges=exclude_dates_list,min_max_development=min_max_development,cohort_dev_df=result_ultimate)
        st.plotly_chart(fig, use_container_width=True, key="forecast_trend_filtered")
    with tab4: 
        st.dataframe(result_ultimate)  
    with tab3:
        
        best_frequencies = select_best_frequency(projected_df, ultimate_freq_avg, result_ultimate,min_max_development=min_max_development,lag=lag)
        
        if 'source' not in best_frequencies.columns:
            best_frequencies['source'] = 'model'

        # --- Load manual overrides from config and apply to DataFrame ---
        manual_overrides = user_config.get("manual_overrides",[])
        allow_manual_overrides = st.sidebar.toggle("allow_manual_overrides",value = True,disabled=True)
        if allow_manual_overrides:
            if config_key in all_configs and 'manual_overrides' in user_config:
                manual_overrides = user_config['manual_overrides']
                for override in manual_overrides:
                    mask = (best_frequencies['year'] == override['year']) & (best_frequencies['month'] == override['month'])
                    best_frequencies.loc[mask, 'best_frequency'] = override['best_frequency']
                    best_frequencies.loc[mask, 'source'] = 'manual'

            # Show editable table for best_frequency
            edited_df = st.data_editor(
                best_frequencies,
                column_config={
                    'best_frequency': st.column_config.NumberColumn('Best Frequency', step=0.0001),
                    'source': st.column_config.TextColumn('Source', disabled=True)
                },
                disabled=[col for col in best_frequencies.columns if col != 'best_frequency'],
                hide_index=True,
                key='best_freq_editor'
            )

            # --- Update manual_overrides based on edits ---
            # 1. Add/update overrides for changed values
            for idx, row in edited_df.iterrows():
                orig_row = best_frequencies.loc[idx]
                if row['best_frequency'] != orig_row['best_frequency']:
                    # Add or update manual override
                    found = False
                    for override in manual_overrides:
                        if override['year'] == row['year'] and override['month'] == row['month']:
                            override['best_frequency'] = float(row['best_frequency'])
                            found = True
                            break
                    if not found:
                        manual_overrides.append({
                            'year': int(row['year']),
                            'month': int(row['month']),
                            'best_frequency': float(row['best_frequency'])
                        })
                    edited_df.at[idx, 'source'] = 'manual'

            # 2. Remove overrides if user deleted via delete button
            st.subheader("Manual Overrides for Best Frequency")
            if manual_overrides:
                manual_df = pd.DataFrame(manual_overrides)
                for i, override in manual_df.iterrows():
                    cols = st.columns([2,2,3,1])
                    cols[0].write(override['year'])
                    cols[1].write(override['month'])
                    cols[2].write(override['best_frequency'])
                    if cols[3].button('Delete', key=f'delete_{i}'):
                        # Remove from manual_overrides
                        manual_overrides = [o for o in manual_overrides if not (o['year'] == override['year'] and o['month'] == override['month'])]
                        # Reset value in edited_df to model value and set source to 'model'
                        mask = (edited_df['year'] == override['year']) & (edited_df['month'] == override['month'])
                        model_val = best_frequencies.loc[mask, 'best_frequency'].values[0]
                        edited_df.loc[mask, 'best_frequency'] = model_val
                        edited_df.loc[mask, 'source'] = 'model'
                        # Update config
                        all_configs[config_key]['manual_overrides'] = manual_overrides

                        safe_write_config(config_path,data=all_configs)
                        # with open(config_path, "w") as f:
                        #     json.dump(all_configs, f, indent=2)
                        st.success(f"Deleted manual override for {override['year']}-{override['month']}")
                        st.rerun()
            else:
                st.write("No manual overrides for this segment/cutoff.")

            all_configs[config_key]['manual_overrides'] = manual_overrides
            safe_write_config(config_path,data=all_configs)
            # with open(config_path, "w") as f:
            #     json.dump(all_configs, f, indent=2)

            fig_best_freq = plot_best_frequency(edited_df, segment)
            st.plotly_chart(fig_best_freq, use_container_width=True, key="best_freq_chart")

        else:
            # --- Always update config with current manual_overrides ---
            all_configs[config_key]['manual_overrides'] = manual_overrides
            safe_write_config(config_path,data=all_configs)
            # with open(config_path, "w") as f:
            #     json.dump(all_configs, f, indent=2)

            st.dataframe(best_frequencies)
            fig_best_freq = plot_best_frequency(best_frequencies, segment)
            st.plotly_chart(fig_best_freq, use_container_width=True, key="best_freq_chart")


# Save config for this segment/cutoff
all_configs[config_key] = {
    "num_claims_threshold": num_claims_threshold,
    "exclude_dates": exclude_dates_list,
    "min_max_development": min_max_development,
    "cutoff_date": str(cutoff_date),
    "min_date": str(min_date),
    "max_date": str(max_date),
    "manual_overrides": manual_overrides
    
}
safe_write_config(config_path,data=all_configs)

# with open(config_path, "w") as f:
#     json.dump(all_configs, f, indent=2)

# Add a new mode to save all segments with their associated configurations
save_all_mode = st.sidebar.checkbox("Save All Segments", value=False)

if save_all_mode:
    if st.sidebar.button("Save All Results"):
        st.sidebar.write("Saving all segments with their configurations...")
        for segment in list_segments:
            st.sidebar.write(f"Processing segment: {segment}")
            #try:
            policies_df_seg = policies_df[policies_df[SEGMENT] == segment]
            claims_df_seg = claims_df[claims_df[SEGMENT]==segment]
            if block == "CSA":
                lag = 0
            else:
                all_configs_lag = fd.load_user_config(config_path_lag)
                config_key_lag = f"{segment}__{str(cutoff_date)}"
                user_config_lag = all_configs_lag.get(config_key_lag, None)
                lag = user_config_lag.get("lag",0)
            analyze_and_save_segment(
                segment,
                cutoff_date,
                all_configs,
                policies_df_seg,
                claims_df_seg,
                date_range,
                granularity,
                False,
                config_path,
                results_path=result_path,
                lag=lag
            )
            st.sidebar.success(f"Results saved successfully for {segment}! 👍")
            # except Exception as e:
            #     st.sidebar.error(f"Error saving results for {segment}: {e} 👎")
            # Use the same cutoff date as the current session
            
            
        save_user_config(config_path, all_configs)
        #st.sidebar.success("All segments saved successfully!")

else:
    # Original save button logic
    save_user_config(config_path, all_configs)
    if st.sidebar.button(f"Save Results for {segment}"):
        try:
            save_data(segment, cutoff_date, best_frequencies,results_path=result_path)
            st.sidebar.success("Results saved successfully! 👍")
        except Exception as e:
            st.sidebar.error(f"Error saving results: {e} 👎")
   
