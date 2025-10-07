
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import numpy as np
import json
import os
import plotly.graph_objects as go
from dateutil.relativedelta import relativedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "helpers"))
from config import config

import frequency_development as fd
from frequency_development.constants import *

from frequency_development.core import load_data

from claim_count_forecast.core import analyze_claims_forecast,plot_best_frequency

# Load paths from configuration
ROOT_FREQUENCY = str(config.get_frequency_results_path()) + "\\"
ROOT_POLICY_FORECAST = str(config.get_policy_results_path()) + "\\"
ROOT_CLAIM_FORECAST = str(config.get_claim_results_path()) + "\\"

config_path = str(config.get_config_path('lag'))

INPUT_BACKUP_MODE_CSA = str(config.BACKUP_MODE_CSA_PATH) + "\\"
INPUT_BACKUP_MODE_TM = str(config.BACKUP_MODE_TM_PATH) + "\\"

result_path = ROOT_CLAIM_FORECAST

scenarios = pd.read_json('scenarios.json')


def calculate_month_difference(start_date_column, end_date_column, row):
    """Compute the month difference between two dates in a row."""
    start_date = row[start_date_column]
    end_date = row[end_date_column]
    if pd.isna(start_date) or pd.isna(end_date):
        return None
    return relativedelta(end_date, start_date).years * 12 + relativedelta(end_date, start_date).months

#@st.cache_data
def get_allowed_frequency_dates(segment:str):
    freq = pd.read_csv(ROOT_FREQUENCY + "best_frequencies.csv")
    return [pd.to_datetime(f).strftime("%Y-%m-%d") for f in freq[freq[SEGMENT]==segment]["cutoff"].unique()]

#@st.cache_data
def get_frequency_per_dep(segment:str,cutoff:str):
    freq = pd.read_csv(ROOT_FREQUENCY + "best_frequencies.csv")
    freq["cutoff"] = pd.to_datetime(freq["cutoff"])
    freq_seg = freq[(freq[SEGMENT]==segment) & (freq["cutoff"]==cutoff)].rename(columns={"best_frequency":"final_weighted_freq"})

    return freq_seg[["cutoff",SEGMENT,DATE_DEPART_END_OF_MONTH,"final_weighted_freq","month","year"]]

#@st.cache_data
def get_allowed_finance_dates(segment:str,cutoff:str):
    pol_dep_ = pd.read_csv(ROOT_POLICY_FORECAST + "pol_count_per_dep_.csv")
    pol_dep_["cutoff"] = pd.to_datetime(pol_dep_["cutoff"])
    cutoff_finance_list = pol_dep_[(pol_dep_[SEGMENT]==segment) & (pol_dep_["cutoff"]==cutoff)]["cutoff_finance"].unique()
    print(cutoff_finance_list)
    return cutoff_finance_list

#@st.cache_data
def get_policy_count_per_dep(segment:str,cutoff:str, cutoff_finance:str):
    pol_dep_ = pd.read_csv(ROOT_POLICY_FORECAST + "pol_count_per_dep_.csv")
    pol_dep_["cutoff"] = pd.to_datetime(pol_dep_["cutoff"])
    pol_dep_["cutoff_finance"] = pd.to_datetime(pol_dep_["cutoff_finance"])
    pol_dep_seg = pol_dep_[(pol_dep_[SEGMENT]==segment) & (pol_dep_["cutoff"]==cutoff) & (pol_dep_["cutoff_finance"]==cutoff_finance)]

    return pol_dep_seg[["cutoff","cutoff_finance",SEGMENT,DATE_DEPART_END_OF_MONTH,"idpol_unique_"]]

def dep_to_received_claims_cohorts(df:pd.DataFrame)->pd.DataFrame:
    """ return the policy volume per departure dates from the training dataframe"""
    #grouped_df = df.groupby(['dateDepart_EndOfMonth',"dateReceived_EndOfMonth"],as_index=False).agg({"idpol_unique":"sum","clmNum_unique":"sum"}).sort_values(['dateDepart_EndOfMonth',"dateReceived_EndOfMonth"])
    grouped_df = df.groupby(['dateDepart_EndOfMonth',"dateReceived_EndOfMonth"],as_index=False).agg({"clmNum_unique":"sum"}).sort_values(['dateDepart_EndOfMonth',"dateReceived_EndOfMonth"])
    for col in ['dateDepart_EndOfMonth',"dateReceived_EndOfMonth"]:
        grouped_df[f"{col}_str"] = grouped_df[col].astype("str")
    
    # represent the past cohorts
    grouped_df["clm_past_present"] = "past"
    return grouped_df


def _save_csv_no_duplicates(filepath, new_df, key_columns):
    """
    Append new_df to filepath, but remove any existing rows with the same key_columns as in new_df.
    """
    if os.path.exists(filepath):
        existing_df = pd.read_csv(filepath)
        existing_df['cutoff'] = pd.to_datetime(existing_df['cutoff'])
        existing_df['cutoff_finance'] = pd.to_datetime(existing_df['cutoff_finance'])
        existing_df['cutoff_frequency'] = pd.to_datetime(existing_df['cutoff_frequency'])

        # Remove rows that match any (segment, cutoff, cutoff_finance) in new_df
        mask = ~existing_df.set_index(key_columns).index.isin(new_df.set_index(key_columns).index)
        combined_df = pd.concat([existing_df[mask], new_df], ignore_index=True)
    else:
        combined_df = new_df
    combined_df.to_csv(filepath, index=False)


def save_data(segment, cutoff_date, cutoff_date_finance, cutoff_date_frequency
              ,final_claims_rec_data, final_claims_dep_data, past_future_claims_cohort_df_for_semantic_model
              , results_path="_results"):
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
     # Prepare best frequencies DataFrame
    r1_df = final_claims_rec_data.copy()
    r2_df = final_claims_dep_data.copy()
    r3_df = past_future_claims_cohort_df_for_semantic_model.copy()

    for df in [r1_df, r2_df, r3_df]:
        df[SEGMENT] = segment
        df["cutoff"] = pd.to_datetime(cutoff_date)
        df["cutoff_finance"] = pd.to_datetime(cutoff_date_finance)
        df["cutoff_frequency"] = pd.to_datetime(cutoff_date_frequency)

    print(os.path.join(results_path, 'final_claims_rec_data.csv'))
    _save_csv_no_duplicates(
        os.path.join(results_path, 'final_claims_rec_data.csv'),
        r1_df,
        [SEGMENT, 'cutoff', 'cutoff_finance','cutoff_frequency']
    )
    
    print(os.path.join(results_path, 'final_claims_dep_data.csv'))
    _save_csv_no_duplicates(
        os.path.join(results_path, 'final_claims_dep_data.csv'),
        r2_df,
        [SEGMENT, 'cutoff', 'cutoff_finance','cutoff_frequency']
    )

    print(os.path.join(results_path, 'final_claims_cohorts_data.csv'))
    _save_csv_no_duplicates(
        os.path.join(results_path, 'final_claims_cohorts_data.csv'),
        r3_df,
        [SEGMENT, 'cutoff', 'cutoff_finance','cutoff_frequency']
    )



# def save_data(segment, cutoff_date, cutoff_date_finance,cutoff_date_frequency
#               , final_claims_rec_data, final_claims_dep_data, past_future_claims_cohort_df_for_semantic_model
#               , results_path="_results"):
#     # Ensure results folder exists
#     if not os.path.exists(results_path):
#         os.makedirs(results_path)
    
    
#     # Prepare best frequencies DataFrame
#     r1_df = final_claims_rec_data.copy()
#     r2_df = final_claims_dep_data.copy()
#     r3_df = past_future_claims_cohort_df_for_semantic_model.copy()
    
#     # Save best frequencies
#     r1 = os.path.join(results_path, 'final_claims_rec_data.csv')
#     r2 = os.path.join(results_path, 'final_claims_dep_data.csv')
#     r3 = os.path.join(results_path, 'final_claims_cohorts_data.csv')

#     def _save(r,df):
#         df["cutoff"] = pd.to_datetime(df["cutoff"])
#         df["cutoff_finance"] = pd.to_datetime(df["cutoff_finance"])
#         df["cutoff_frequency"] = pd.to_datetime(df["cutoff_frequency"])
#         if os.path.exists(r):
#             existing_r = pd.read_csv(r)
#             existing_r["cutoff"] = pd.to_datetime(existing_r["cutoff"])
#             existing_r["cutoff_finance"] = pd.to_datetime(existing_r["cutoff_finance"])
#             existing_r["cutoff_frequency"] = pd.to_datetime(existing_r["cutoff_frequency"])
#             mask = ~(
#                 (existing_r[SEGMENT] == segment) &
#                 (existing_r['cutoff'] == pd.to_datetime(cutoff_date)) & 
#                 (existing_r['cutoff_finance'] == pd.to_datetime(cutoff_date_finance)) & 
#                 (existing_r['cutoff_frequency'] == pd.to_datetime(cutoff_date_frequency))
#             )
#             existing_r = existing_r[mask]
#             existing_r = pd.concat([existing_r, df], ignore_index=True)
#         else:
#             existing_r = df
#         existing_r.to_csv(r, index=False)

#     _save(r1,r1_df)
#     _save(r2,r2_df)
#     _save(r3,r3_df)






##### start application
st.set_page_config(layout="wide")



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


# get frequency scenario
# get policy count scenario
# get claims data


st.sidebar.title("Controls")

situation = st.sidebar.selectbox("situation under analysis",scenarios.keys(),index=0)
st.sidebar.text(f"situation: {situation}")
scenario_for_situation = scenarios[situation]

scenario = st.sidebar.selectbox("scenario selected",scenario_for_situation.keys(),index=0)

scenario_dates = scenario_for_situation[scenario]

st.sidebar.write(scenario_dates)
cutoff_date = scenario_dates['cutoff']
cutoff_date_frequency = scenario_dates['cutoff_frequency']
cutoff_date_finance = scenario_dates['cutoff_finance']


block = st.sidebar.selectbox("CSA/TM",["CSA","TM"],index=0)

if block == "CSA":
    backup = st.sidebar.toggle("backup mode",value=True)
else:
    backup = True #true as default for Trip Mate

if block == "CSA":
    if backup:
        # allowed_dates_backup = []
        # # list input folders for finance data
        # date_folders = [
        #     name for name in os.listdir(INPUT_BACKUP_MODE_CSA)
        #     if os.path.isdir(os.path.join(INPUT_BACKUP_MODE_CSA,name))
        # ]

        # for folder in date_folders:
        #     try:
        #         date = pd.to_datetime(folder, format='%Y-%m-%d',errors='raise')
        #         allowed_dates_backup.append(date)
        #     except Exception:
        #         pass

        # allowed_dates_backup_str = [d.strftime('%Y-%m-%d') for d in sorted(allowed_dates_backup)]
        # cutoff_date = st.sidebar.selectbox(
        #     "Select Cutoff Date (for forecast projection)",
        #     ["allowed_dates_backup_str"],
        #     index = 0,
        #     disabled = False
        # )
        policies_df, claims_df = load_data_backup(cutoff_date)
    else:
        
        # allowed_dates =[
        #     pd.Timestamp('2025-04-30'),
        # ]

        # allowed_dates_str = [d.strftime('%Y-%m-%d') for d in allowed_dates]

        # cutoff_date = st.sidebar.selectbox(
        #     "Select Cutoff Date (for forecast projection)",
        #     allowed_dates_str,
        #     index = 0,
        #     disabled = True
        # )
        policies_df, claims_df = load_data(cutoff_date)
    

elif block =="TM":

    # allowed_dates_backup = []
    #     # list input folders for finance data
    # date_folders = [
    #     name for name in os.listdir(INPUT_BACKUP_MODE_TM)
    #     if os.path.isdir(os.path.join(INPUT_BACKUP_MODE_TM,name))
    # ]
    # print(date_folders)
    # for folder in date_folders:
    #     try:
    #         date = pd.to_datetime(folder, format='%Y-%m-%d',errors='raise')
    #         allowed_dates_backup.append(date)
    #     except Exception:
    #         pass

    # allowed_dates_backup_str = [d.strftime('%Y-%m-%d') for d in sorted(allowed_dates_backup)]
    # cutoff_date = st.sidebar.selectbox(
    #     "Select Cutoff Date (for forecast projection)",
    #     allowed_dates_backup_str,
    #     index = 0,
    #     disabled = False
    # )

    policies_df, claims_df = load_data_backup_tripmate(cutoff_date)


existing_df = pd.read_csv(os.path.join(result_path, 'final_claims_rec_data.csv'))
existing_df['cutoff'] = pd.to_datetime(existing_df['cutoff'] )
existing_df['cutoff_finance'] = pd.to_datetime(existing_df['cutoff_finance'])
existing_df['cutoff_frequency'] = pd.to_datetime(existing_df['cutoff_frequency'])

if not existing_df.empty:
    st.sidebar.write("Already saved results:")
    segment_saved_table = existing_df[(existing_df["cutoff"]==pd.to_datetime(cutoff_date)) 
                                      & (existing_df["cutoff_finance"]==pd.to_datetime(cutoff_date_finance))
                                      & (existing_df["cutoff_frequency"]==pd.to_datetime(cutoff_date_frequency))
                                      ][[SEGMENT, 'cutoff', 'cutoff_finance','cutoff_frequency']].drop_duplicates().sort_values(by=SEGMENT)
    st.sidebar.dataframe(segment_saved_table)
    already_saved_segments = sorted(segment_saved_table[SEGMENT].unique())
    #st.sidebar.write(already_saved_segments)
else:
    st.sidebar.write("No results saved yet.")

show_only_non_saved_segments = st.sidebar.toggle("only show non saved segment",value=True)

irrelevant = ['Unknown','Timeshare','Expedia',"Tripmate","TripMate","Identity Theft"]
list_segments= claims_df[~claims_df[SEGMENT].isin(irrelevant)][SEGMENT].unique()


list_segments= sorted(claims_df[~claims_df[SEGMENT].isin(irrelevant)][SEGMENT].unique())
if show_only_non_saved_segments:
    list_segments = sorted(list(set(list_segments) - set(already_saved_segments)))

if list_segments == []:
    st.sidebar.warning("No more segments to work on")
else:
    segment = st.sidebar.selectbox('Select Segment', sorted(list_segments))

    # cutoff_date_frequency = st.sidebar.selectbox(
    #     "Select Cutoff Date (for frequency selection)",
    #     get_allowed_frequency_dates(segment),
    #     index = 0,
    #     disabled = False
    # )

    # allowed_dates_finance_str = get_allowed_finance_dates(segment=segment,cutoff=cutoff_date)

    # if allowed_dates_finance_str:
    #     cutoff_date_finance = st.sidebar.selectbox(
    #         "Select Cutoff Date (for finance assumption)",
    #         allowed_dates_finance_str,
    #         index = 0,
    #         disabled = False
    #     )
    # else:
    #     st.warning(f"No scenario found for {cutoff_date} for {segment}. Please forecast policy count.")


    st.title("Claim Count Forecast")
    #select frequency
    frequency_from_file_df_ = get_frequency_per_dep(segment=segment,cutoff=cutoff_date_frequency)
    
    if block == "CSA":
        lag = 0
    else:
        all_configs = fd.load_user_config(config_path)
        config_key = f"{segment}__{str(cutoff_date)}"
        user_config = all_configs.get(config_key, None)
        lag = user_config.get("lag",0)

    lag = st.sidebar.number_input("lag",value=lag,disabled=True,help="as defined in the policy count forecast module")

    past_future_pol_cohorts_ = get_policy_count_per_dep(segment=segment,cutoff = cutoff_date, cutoff_finance=cutoff_date_finance)


    results_dict = analyze_claims_forecast(
                    segment,
                    cutoff_date,
                    cutoff_date_finance,
                    cutoff_date_frequency,
                    policies_df,
                    claims_df,
                    past_future_pol_cohorts_,
                    frequency_from_file_df_,
                    lag
                    )

    fig_dev_patterns = results_dict["fig_dev_patterns"]
    past_future_claims_cohort_df_ = results_dict["past_future_claims_cohort_df_"]
    final_claims_rec_data = results_dict["final_claims_rec_data"]
    past_future_claims_cohort_df_for_semantic_model = results_dict["past_future_claims_cohort_df_for_semantic_model"]
    final_claims_dep_data = results_dict["final_claims_dep_data"]
    claims_df_seg = results_dict["claims_df_seg"]

    # claims_df_seg = claims_df[claims_df[SEGMENT]==segment]
    # claims_df_seg = claims_df_seg.groupby([DEPARTURE_DAY,RECEIVED_DAY,DATE_DEPART_END_OF_MONTH,DATE_RECEIVED_END_OF_MONTH]).agg({CLAIM_COUNT:"sum"}).reset_index()

    # claims_df_seg["depart_to_receive_month"] = claims_df_seg.apply(lambda row: calculate_month_difference(DEPARTURE_DAY, RECEIVED_DAY, row), axis=1)
    # claims_df_seg["clmNum_unique"] = claims_df_seg[CLAIM_COUNT]
    # claims_df_seg["dateDepart"] = pd.to_datetime(claims_df_seg["dateDepart"])
    # claims_df_seg["dateReceived"] = pd.to_datetime(claims_df_seg["dateReceived"])
    # claims_df_seg[DATE_DEPART_END_OF_MONTH] = pd.to_datetime(claims_df_seg[DATE_DEPART_END_OF_MONTH])
    # claims_df_seg[DATE_RECEIVED_END_OF_MONTH] = pd.to_datetime(claims_df_seg[DATE_RECEIVED_END_OF_MONTH])



    # past_future_pol_cohorts_["dateDepart_EndOfMonth"] = pd.to_datetime(past_future_pol_cohorts_["dateDepart_EndOfMonth"])
    # frequency_from_file_df_["dateDepart_EndOfMonth"] = pd.to_datetime(frequency_from_file_df_["dateDepart_EndOfMonth"])

    # past_future_pol_cohorts_df_with_freq = pd.merge(past_future_pol_cohorts_,frequency_from_file_df_[["dateDepart_EndOfMonth","final_weighted_freq"]],how="left",on="dateDepart_EndOfMonth")

    # # get the claim count per cohort
    # past_future_pol_cohorts_df_with_freq["clmNum_unique_"] = past_future_pol_cohorts_df_with_freq["idpol_unique_"] * past_future_pol_cohorts_df_with_freq["final_weighted_freq"]

    # # st.subheader("past_future_pol_cohorts_df_with_freq")
    # # st.dataframe(past_future_pol_cohorts_df_with_freq)

    # # Step 4: find development pattern
    # month_diff_column_name = "depart_to_receive_month"
    # #method_dep_to_rec = "year"
    # level = 0.95
    # metric_column_name="clmNum_unique"
    # metric_column_name_output = "clmNum_unique_"
    # start_date_column = "dateDepart"
    # end_date_column = "dateReceived"
    # past_futurename = "clm_past_present"
    # method_dep_to_rec = "month"
    # df_copy = claims_df_seg.copy()
    # training_df = claims_df_seg.copy()
    # extraction_date = cutoff_date


    # monthly_dev_patterns, selected_pattern,dynamic_cols,dynamic_cols_end = select_main_development_info(df_copy
    #                                                             ,start_date_column = start_date_column
    #                                                             ,end_date_column = end_date_column
    #                                                             ,month_diff_column_name=month_diff_column_name
    #                                                             ,metric_column_name=metric_column_name
    #                                                             ,level=level
    #                                                             ,method=method_dep_to_rec)

    # fig_dev_patterns = plot_development_patterns(monthly_dev_patterns
    #                                     ,selected_pattern
    #                                     ,month_diff_column_name
    #                                     ,dynamic_cols
    #                                     ,end_date=extraction_date
    #                                     ,xaxis_title="Departure to Receive Month"
    #                                     ,name_agg_line='Overall Development (avg by claim volume)'
    #                                     ,method=method_dep_to_rec
    #                                     ,show = False)

    # selected_development = _select_development_to_forecast(selected_pattern,month_diff_column_name,dynamic_cols,method_dep_to_rec)

    # # Step 5: forecast future claims from dep to receive
    # future_claims_cohort_df_ = _develop_future_cohorts(past_future_pol_cohorts_df_with_freq
    #                             ,selected_development
    #                             ,month_diff_column_name
    #                             ,method = method_dep_to_rec
    #                             ,start_date_eom = dynamic_cols["start_date_eom"]
    #                             ,start_date_month = dynamic_cols["start_date_month"]
    #                             ,end_date_eom = dynamic_cols_end["end_date_eom"]
    #                             ,metric_column_name=metric_column_name_output
    #                             ,past_futurename = past_futurename
    #                             ,metric_column_name_output=metric_column_name_output
    #                             ,additional_group_col=None)

    # # st.subheader("future_claims_cohort_df_")
    # # st.dataframe(future_claims_cohort_df_)
    # # Step 6: get clean past cohorts:
    # past_claims_cohort_df = dep_to_received_claims_cohorts(training_df[training_df["dateReceived"]<=extraction_date])

    # # fig_plot_dep_rec_twolines_past_cohorts = plot_dep_to_received_claims_twolines_cohorts(past_claims_cohort_df)
    # past_claims_cohort_df["dateDepart_EndOfMonth"] = pd.to_datetime(past_claims_cohort_df["dateDepart_EndOfMonth"])
    # past_claims_cohort_df["dateReceived_EndOfMonth"] = pd.to_datetime(past_claims_cohort_df["dateReceived_EndOfMonth"])

    # future_claims_cohort_df_["dateDepart_EndOfMonth"] = pd.to_datetime(future_claims_cohort_df_["dateDepart_EndOfMonth"])
    # future_claims_cohort_df_["dateReceived_EndOfMonth"] = pd.to_datetime(future_claims_cohort_df_["dateReceived_EndOfMonth"])

    # # Step 6: get final cohorts to compute non reported yet claims
    # final_claims_cohorts_ = future_claims_cohort_df_[["dateDepart_EndOfMonth","dateReceived_EndOfMonth","clmNum_unique_"]].merge(past_claims_cohort_df[["dateDepart_EndOfMonth","dateReceived_EndOfMonth","clmNum_unique"]]
    #                                                                                                                         ,how="outer"
    #                                                                                                                         ,on=["dateDepart_EndOfMonth","dateReceived_EndOfMonth"])

    # if lag > 0:
    #     end_valid_date = pd.to_datetime(extraction_date) + pd.offsets.MonthEnd(-lag)
    # else:
    #     end_valid_date = pd.to_datetime(extraction_date)

    # # Corner Case #1: past cohorts with future received claims
    # nyi_condition_1 = (final_claims_cohorts_["clmNum_unique"].isna())
    # nyi_condition_2 = (final_claims_cohorts_["dateDepart_EndOfMonth"]<=extraction_date)
    # nyi_condition_3 = (final_claims_cohorts_["dateReceived_EndOfMonth"]>extraction_date)
    # #nyi_condition_2 = (final_claims_cohorts_["dateDepart_EndOfMonth"]<=end_valid_date)
    # #nyi_condition_3 = (final_claims_cohorts_["dateReceived_EndOfMonth"]>end_valid_date)
    # not_yet_incurred = final_claims_cohorts_[(nyi_condition_1) & (nyi_condition_2) & (nyi_condition_3)]


    # past_claims_cohort_df_with_not_yet_incurred = pd.concat([past_claims_cohort_df[["dateDepart_EndOfMonth","dateReceived_EndOfMonth","clmNum_unique"]],
    #                                                         not_yet_incurred[["dateDepart_EndOfMonth","dateReceived_EndOfMonth","clmNum_unique_"]]],axis=0)

    # to_develop_not_fully_dev_claims_cohorts = final_claims_cohorts_.groupby(["dateDepart_EndOfMonth"],as_index=False).agg({"clmNum_unique_":"sum"})
    # past_claims_cohort_df_with_not_yet_incurred = pd.merge(past_claims_cohort_df_with_not_yet_incurred,to_develop_not_fully_dev_claims_cohorts,how="left",on="dateDepart_EndOfMonth",suffixes=["","target"])

    # # Calculate the remaining amount needed for each 'dateDepart_EndOfMonth' to reach the target
    # past_claims_cohort_df_with_not_yet_incurred['current_sum'] = past_claims_cohort_df_with_not_yet_incurred.groupby('dateDepart_EndOfMonth')['clmNum_unique'].transform('sum')
    # past_claims_cohort_df_with_not_yet_incurred['expected_sum'] = past_claims_cohort_df_with_not_yet_incurred.groupby('dateDepart_EndOfMonth')['clmNum_unique_'].transform('sum')

    # past_claims_cohort_df_with_not_yet_incurred['remaining_to_target'] = np.where(past_claims_cohort_df_with_not_yet_incurred['clmNum_unique_target'] >= past_claims_cohort_df_with_not_yet_incurred['current_sum'],
    #                                                                                 past_claims_cohort_df_with_not_yet_incurred['clmNum_unique_target'] - past_claims_cohort_df_with_not_yet_incurred['current_sum'],
    #                                                                                 0)

    # # past_claims_cohort_df_with_not_yet_incurred['clmNum_unique_'] = np.where(
    # #                                                                 past_claims_cohort_df_with_not_yet_incurred['remaining_to_target'] >= 0,
    # #                                                                 past_claims_cohort_df_with_not_yet_incurred['clmNum_unique_'] * past_claims_cohort_df_with_not_yet_incurred['remaining_to_target'] / past_claims_cohort_df_with_not_yet_incurred['expected_sum'],
    # #                                                                 0
    # #                                                                 )
    # past_claims_cohort_df_with_not_yet_incurred['clmNum_unique_'] = past_claims_cohort_df_with_not_yet_incurred['clmNum_unique_'] * past_claims_cohort_df_with_not_yet_incurred['remaining_to_target'] / past_claims_cohort_df_with_not_yet_incurred['expected_sum']

    # # I don't want to create negative claims
    # #past_claims_cohort_df_with_not_yet_incurred.loc[past_claims_cohort_df_with_not_yet_incurred['remaining_to_target'] < 0, 'clmNum_unique_'] = 0

    # # Fill NaN in 'clmNum_unique_' with values from 'clmNum_unique'
    # past_claims_cohort_df_with_not_yet_incurred["clmNum_unique_"] = past_claims_cohort_df_with_not_yet_incurred["clmNum_unique_"].fillna(past_claims_cohort_df_with_not_yet_incurred["clmNum_unique"])

    # past_claims_cohort_df_with_not_yet_incurred["clm_past_present"] = "past"

    # # CORNER CASE #2
    # bttf_condition_1 = ~(final_claims_cohorts_["clmNum_unique"].isna())
    # bttf_condition_2 = ~(final_claims_cohorts_["clmNum_unique_"].isna())
    # #bttf_condition_3 = (final_claims_cohorts_["dateDepart_EndOfMonth"]>extraction_date)
    # #btff_condition_4 = (final_claims_cohorts_["dateReceived_EndOfMonth"]<=extraction_date)
    # bttf_condition_3 = (final_claims_cohorts_["dateDepart_EndOfMonth"]>end_valid_date)
    # btff_condition_4 = (final_claims_cohorts_["dateReceived_EndOfMonth"]<=end_valid_date)

    # not_yet_departed_already_incurred = final_claims_cohorts_[(bttf_condition_1) &(bttf_condition_3) & (btff_condition_4)]

    # not_yet_departed_not_yet_incurred = final_claims_cohorts_[(bttf_condition_3) & (~btff_condition_4) & (bttf_condition_2)]

    # future_claims_cohorts_with_not_yet_incurred = pd.concat([not_yet_departed_already_incurred[["dateDepart_EndOfMonth","dateReceived_EndOfMonth","clmNum_unique"]],
    #                                                         not_yet_departed_not_yet_incurred[["dateDepart_EndOfMonth","dateReceived_EndOfMonth","clmNum_unique_"]]],axis=0)

    # future_claims_cohorts_with_not_yet_incurred_target = pd.merge(future_claims_cohorts_with_not_yet_incurred,to_develop_not_fully_dev_claims_cohorts,how="left",on="dateDepart_EndOfMonth",suffixes=["","target"])
    # future_claims_cohorts_with_not_yet_incurred_target['current_sum'] = future_claims_cohorts_with_not_yet_incurred_target.groupby('dateDepart_EndOfMonth')['clmNum_unique'].transform('sum')
    # future_claims_cohorts_with_not_yet_incurred_target['expected_sum'] = future_claims_cohorts_with_not_yet_incurred_target.groupby('dateDepart_EndOfMonth')['clmNum_unique_'].transform('sum')
    # # don't create negative claims to reach the target
    # future_claims_cohorts_with_not_yet_incurred_target['remaining_to_target'] = np.where(future_claims_cohorts_with_not_yet_incurred_target['clmNum_unique_target'] >= future_claims_cohorts_with_not_yet_incurred_target['current_sum'],
    #                                                                                 future_claims_cohorts_with_not_yet_incurred_target['clmNum_unique_target'] - future_claims_cohorts_with_not_yet_incurred_target['current_sum'],
    #                                                                                 0)
    # future_claims_cohorts_with_not_yet_incurred_target['clmNum_unique_'] = future_claims_cohorts_with_not_yet_incurred_target['clmNum_unique_'] * future_claims_cohorts_with_not_yet_incurred_target['remaining_to_target'] / future_claims_cohorts_with_not_yet_incurred_target['expected_sum']

    # future_claims_cohorts_with_not_yet_incurred_target["clmNum_unique_"] = future_claims_cohorts_with_not_yet_incurred_target["clmNum_unique_"].fillna(future_claims_cohorts_with_not_yet_incurred_target["clmNum_unique"])


    # future_claims_cohorts_with_not_yet_incurred_target["clm_past_present"] = "future"
    # # Use numpy.where to efficiently check each date and assign "future" or "past"
    # #future_claims_cohorts_with_not_yet_incurred_target["clm_past_present"] = np.where(future_claims_cohorts_with_not_yet_incurred_target["dateDepart_EndOfMonth"] > extraction_date, "future", "past")

    # # past_claims_cohort_df_with_not_yet_incurred["clm_past_present"] = "past"

    # # Step 7: remove not needed cohorts from future
    # #future_claims_cohorts_with_not_yet_incurred_target = future_claims_cohorts_with_not_yet_incurred_target[future_claims_cohorts_with_not_yet_incurred_target["dateReceived_EndOfMonth"]>extraction_date]

    # # keep the rows where we did not already observed a received claim
    # future_claims_cohorts_with_not_yet_incurred_target = future_claims_cohorts_with_not_yet_incurred_target[(future_claims_cohorts_with_not_yet_incurred_target["clmNum_unique"].isna())]

    # # Step 8: Concat the end result
    # past_future_claims_cohort_df_ = pd.concat([past_claims_cohort_df_with_not_yet_incurred[[past_futurename,"dateDepart_EndOfMonth","dateReceived_EndOfMonth",metric_column_name_output]]
    #                                     ,future_claims_cohorts_with_not_yet_incurred_target[[past_futurename,"dateDepart_EndOfMonth","dateReceived_EndOfMonth",metric_column_name_output]]])

    # # if there's a lag, we need to remove the claims that are duplicated
    # # we want to keep the expected claims coming from "future"
    # # Find duplicated 'dateReceived_EndOfMonth' values (keeping all occurrences for further inspection)
    # duplicated = past_future_claims_cohort_df_.duplicated(subset=["dateDepart_EndOfMonth",'dateReceived_EndOfMonth'], keep=False)

    # # Filter for rows that are duplicated and have 'clm_past_present' marked as 'past'
    # rows_to_drop = past_future_claims_cohort_df_[duplicated & (past_future_claims_cohort_df_['clm_past_present'] == 'past')]

    # # Drop these rows from the DataFrame
    # past_future_claims_cohort_df_ = past_future_claims_cohort_df_.drop(rows_to_drop.index)

    # # Use numpy.where to efficiently check each date and assign "future" or "past"
    # past_future_claims_cohort_df_["clm_past_present"] = np.where(past_future_claims_cohort_df_["dateDepart_EndOfMonth"] > extraction_date, "future", "past")

    # #past_future_claims_cohort_df_ = past_future_claims_cohort_df_[past_future_claims_cohort_df_["dateReceived_EndOfMonth"]>extraction_date]
    # # Corner Case #2: Future Cohort with received claims in the past:

    # # remove the estimated received before the end_date_valuation
    # # if config_file.get("lag",0) > 0:
    # #     end_valid_date = pd.to_datetime(extraction_date) + pd.offsets.MonthEnd(-config_file.get("lag",0))
    # # else:
    # #     end_valid_date = pd.to_datetime(extraction_date)

    # # past_future_claims_cohort_df_.loc[(past_future_claims_cohort_df_["dateReceived_EndOfMonth"]<=end_valid_date) & (past_future_claims_cohort_df_[past_futurename]=="future"),metric_column_name_output] = 0


    # for col in ["dateDepart_EndOfMonth","dateReceived_EndOfMonth"]:
    #     past_future_claims_cohort_df_[f"{col}_str"] = past_future_claims_cohort_df_[col].astype("str")

    # for col in ["dateDepart","dateReceived"]:
    #     past_future_claims_cohort_df_[col + '_year'] = pd.to_datetime(past_future_claims_cohort_df_[col + '_EndOfMonth']).dt.year
    #     past_future_claims_cohort_df_[col + '_month'] = pd.to_datetime(past_future_claims_cohort_df_[col + '_EndOfMonth']).dt.month








    tab0, tab1 = st.tabs(["Final tables for semantic model","Input tables for Claim Count Model"])



    with tab0:
        st.markdown(f""" 
                    Using input tables: 
                    - financial assumption as of {cutoff_date_finance}
                    - historical data as of {cutoff_date}
                    - frequency data as of {cutoff_date_frequency}
                    """)
        tab00,tab01,tab02,tab03,tab04 = st.tabs(["Cohorts","frequency","pol count forecast (departure)","Forecast Per Received Date (main output)","Forecast Per Departure Date"])

        
        with tab03:
            st.subheader(f"Claim Count Forecast Per Received Date for {segment}")
            start_x_axis= max(pd.to_datetime("2021-01-01"),past_future_claims_cohort_df_[DATE_RECEIVED_END_OF_MONTH].min())
            end_x_axis = min(pd.to_datetime("2027-12-31"),past_future_claims_cohort_df_[DATE_RECEIVED_END_OF_MONTH].max())

            # final_claims_rec_data = past_future_claims_cohort_df_[past_future_claims_cohort_df_[DATE_RECEIVED_END_OF_MONTH]>=start_x_axis].groupby([DATE_RECEIVED_END_OF_MONTH],as_index=False).agg({"clmNum_unique_":"sum"})
            # final_claims_rec_data[SEGMENT] = segment
            # final_claims_rec_data["cutoff"] = pd.to_datetime(cutoff_date)
            # final_claims_rec_data["cutoff_finance"] = pd.to_datetime(cutoff_date_finance)
            # final_claims_rec_data["cutoff_frequency"] = pd.to_datetime(cutoff_date_frequency)
            # final_claims_rec_data = final_claims_rec_data[["cutoff","cutoff_finance","cutoff_frequency",SEGMENT,DATE_RECEIVED_END_OF_MONTH,"clmNum_unique_"]]

            fig_final_claims_rec_date = px.line(final_claims_rec_data,x="dateReceived_EndOfMonth",y="clmNum_unique_")
            st.plotly_chart(fig_final_claims_rec_date, use_container_width=True, key="fig_final_claims_rec_date")
            st.dataframe(final_claims_rec_data)        

        with tab00:
            st.subheader(f"Claim Count Forecasted Cohorts for {segment} as of {cutoff_date}")
            # past_future_claims_cohort_df_[SEGMENT] = segment
            # past_future_claims_cohort_df_["cutoff"] = pd.to_datetime(cutoff_date)
            # past_future_claims_cohort_df_["cutoff_finance"] = pd.to_datetime(cutoff_date_finance)
            # past_future_claims_cohort_df_["cutoff_frequency"] = pd.to_datetime(cutoff_date_frequency)
            
            # past_future_claims_cohort_df_for_semantic_model = past_future_claims_cohort_df_[["cutoff","cutoff_finance","cutoff_frequency",SEGMENT,"clm_past_present",DATE_DEPART_END_OF_MONTH,DATE_RECEIVED_END_OF_MONTH,"clmNum_unique_"]]
            st.dataframe(past_future_claims_cohort_df_for_semantic_model)
        
        with tab04:
            st.subheader(f"Claim Count Forecast Per Departure Date for {segment}")
            start_x_axis= max(pd.to_datetime("2021-01-01"),past_future_claims_cohort_df_[DATE_RECEIVED_END_OF_MONTH].min())
            # final_claims_dep_data = past_future_claims_cohort_df_[past_future_claims_cohort_df_[DATE_DEPART_END_OF_MONTH]>=start_x_axis].groupby([DATE_DEPART_END_OF_MONTH],as_index=False).agg({"clmNum_unique_":"sum"})
            # final_claims_dep_data[SEGMENT] = segment
            # final_claims_dep_data["cutoff"] = pd.to_datetime(cutoff_date)
            # final_claims_dep_data["cutoff_finance"] = pd.to_datetime(cutoff_date_finance)
            # final_claims_dep_data["cutoff_frequency"] = pd.to_datetime(cutoff_date_frequency)

            # final_claims_dep_data = final_claims_dep_data[["cutoff","cutoff_finance","cutoff_frequency",SEGMENT,DATE_DEPART_END_OF_MONTH,"clmNum_unique_"]]

            fig_final_claims_dep_date = px.line(final_claims_dep_data,x="dateDepart_EndOfMonth",y="clmNum_unique_")

            st.plotly_chart(fig_final_claims_dep_date, use_container_width=True, key="fig_final_claims_dep_date")
            st.dataframe(final_claims_dep_data)

        with tab01: 
            st.dataframe(frequency_from_file_df_)
            fig_best_freq = plot_best_frequency(frequency_from_file_df_, segment,y='final_weighted_freq')
            st.plotly_chart(fig_best_freq, use_container_width=True, key="best_freq_chart")

        with tab02:
            st.subheader(f"Policy Count Forecast as of {cutoff_date} using financial assumption as of {cutoff_date_finance}")
            st.dataframe(past_future_pol_cohorts_)


    with tab1:
        tab11,tab12,tab13,tab14 = st.tabs(["frequency","pol count forecast (departure)","historical claim count","development pattern"])
        with tab11: 
            st.dataframe(frequency_from_file_df_)
            fig_best_freq = plot_best_frequency(frequency_from_file_df_, segment,y='final_weighted_freq')
            st.plotly_chart(fig_best_freq, use_container_width=True, key="best_freq_chart")

        with tab12:
            st.subheader(f"Policy Count Forecast as of {cutoff_date} using financial assumption as of {cutoff_date_finance}")
            st.dataframe(past_future_pol_cohorts_)

        with tab13: 
            st.dataframe(claims_df_seg)

        with tab14:
            st.plotly_chart(fig_dev_patterns, use_container_width=True, key="fig_dev_patterns")

    # Add a new mode to save all segments with their associated configurations
    save_all_mode = st.sidebar.checkbox("Save All Segments", value=False)
    if save_all_mode:
        if st.sidebar.button("Save All Results"):
            st.sidebar.write("Saving all segments with their configurations...")
            for segment in list_segments:
                st.sidebar.write(f"Processing segment: {segment}")
                try:
                    frequency_from_file_df_ = get_frequency_per_dep(segment=segment,cutoff=cutoff_date_frequency)
                    if block == "CSA":
                        lag = 0
                    else:
                        all_configs = fd.load_user_config(config_path)
                        config_key = f"{segment}__{str(cutoff_date)}"
                        user_config = all_configs.get(config_key, None)
                        lag = user_config.get("lag",0)

                    past_future_pol_cohorts_ = get_policy_count_per_dep(segment=segment,cutoff = cutoff_date, cutoff_finance=cutoff_date_finance)


                    results_dict = analyze_claims_forecast(
                                    segment,
                                    cutoff_date,
                                    cutoff_date_finance,
                                    cutoff_date_frequency,
                                    policies_df,
                                    claims_df,
                                    past_future_pol_cohorts_,
                                    frequency_from_file_df_,
                                    lag
                                    )

                    fig_dev_patterns = results_dict["fig_dev_patterns"]
                    past_future_claims_cohort_df_ = results_dict["past_future_claims_cohort_df_"]
                    final_claims_rec_data = results_dict["final_claims_rec_data"]
                    past_future_claims_cohort_df_for_semantic_model = results_dict["past_future_claims_cohort_df_for_semantic_model"]
                    final_claims_dep_data = results_dict["final_claims_dep_data"]
                    claims_df_seg = results_dict["claims_df_seg"]
                    
                    save_data(segment, cutoff_date, cutoff_date_finance,cutoff_date_frequency
                    , final_claims_rec_data, final_claims_dep_data, past_future_claims_cohort_df_for_semantic_model
                    , results_path=result_path)
                    st.sidebar.success(f"Results saved successfully for {segment}! 👍")
                except Exception as e:
                    st.sidebar.error(f"Error saving results for {segment}: {e} 👎")
                # Use the same cutoff date as the current session
                
            #save_user_config(config_path, all_configs)
            #st.sidebar.success("All segments saved successfully!")

    else:
        # Original save button logic
        #save_user_config(config_path, all_configs)
        if st.sidebar.button(f"Save Results for {segment}"):
            #try:
            st.dataframe(final_claims_rec_data.head(3))
            st.dataframe(final_claims_dep_data.head(3))
            st.dataframe(past_future_claims_cohort_df_for_semantic_model.head(3))

            save_data(segment, cutoff_date, cutoff_date_finance,cutoff_date_frequency
            , final_claims_rec_data, final_claims_dep_data, past_future_claims_cohort_df_for_semantic_model
            , results_path=result_path)
            st.sidebar.success("Results saved successfully! 👍")
            #except Exception as e:
            #    st.sidebar.error(f"Error saving results: {e} 👎")