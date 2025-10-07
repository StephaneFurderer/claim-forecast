from policy_count_forecast import select_main_development_info
import pandas as pd
import numpy as np
import plotly.express as px
from frequency_development.constants import *
from dateutil.relativedelta import relativedelta
import plotly.graph_objects as go

from policy_count_forecast import select_main_development_info,_select_development_to_forecast

def plot_development_patterns(development_patterns:pd.DataFrame
                              , selected_pattern:pd.DataFrame
                              , month_diff_column_name:str
                              , dynamic_cols:dict
                              , end_date:str
                              , method:str = "year"
                              , xaxis_title:str='Departure to Received Months'
                              , name_agg_line:str = 'Overall Distribution'
                              , show:bool=False):
    
    # Extract dynamic column names from kwargs or default to a preset if not provided
    start_date_eom = dynamic_cols['start_date_eom'] #, 'dateDepart_EndOfMonth')  # Fallback to 'dateDepart_EndOfMonth' if not specified
    start_date_year = dynamic_cols['start_date_year'] #, 'dateDepart_year')  # Fallback to 'dateDepart_year' if not specified
    start_date_month = dynamic_cols['start_date_month'] #, 'dateDepart_month')  # Fallback to 'dateDepart_month' if not specified
    
    # Group the data
    grouped = development_patterns[development_patterns[start_date_eom]<=end_date].groupby([start_date_year, start_date_month])

    # Calculate the no. of groups to dynamically set the color gradient
    num_groups = len(grouped)

    # Grayscale gradient: light (closer to 255) to dark (closer to 0)
    grayscale_values = np.linspace(200, 50, num_groups)  # Using 200-50 to avoid too-light/dark

    fig = go.Figure()

    for i, ((year, month), group) in enumerate(grouped):
        gray_value = int(grayscale_values[i])
        color = f'rgba({gray_value}, {gray_value}, {gray_value}, 1.0)'  # Same RGB for grayscale

        fig.add_trace(go.Scatter(
            x=group[month_diff_column_name],
            y=group['cumulative_probability'],
            mode='lines+markers',
            name=f'Year {year}, Month {month:02d}',
            line=dict(color=color)
        ))

     # Add the overall distribution lines
    if method == "year":
        fig.add_trace(go.Scatter(
            x=selected_pattern[month_diff_column_name],
            y=selected_pattern['cumulative_probability'],
            mode='lines+text',
            name=name_agg_line,
            line=dict(color='red', width=2),
            text=selected_pattern['percentage_text'], textposition="top center"
        ))

    elif method == "month":
        # Prepare slider steps
        
        # Loop through each month for the overall distribution
        for month, group in selected_pattern.groupby(dynamic_cols['start_date_month']):
            fig.add_trace(go.Scatter(
                x=group[month_diff_column_name],
                y=group['cumulative_probability'],
                mode='lines+text',
                name=f'{name_agg_line} - Month {month}',
                line=dict(color='red', width=2),
                text=group['percentage_text'], textposition="top center"
            ))


    fig.update_layout(
        title='Cumulative Probability by Year and Month',
        xaxis_title=xaxis_title,
        yaxis_title='Cumulative Probability',
        legend_title='Year, Month',
        yaxis=dict(tickformat=".2%"),
        plot_bgcolor="white",  # Adjust plot background if necessary
        #paper_bgcolor='rgba(0,0,0,0)',  # Adjust paper background if necessary
    )

    if show:
        fig.show()
    return fig


def plot_best_frequency(best_frequencies: pd.DataFrame, segment: str, y:str = 'best_frequency'):
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
    fig.update_layout(
        showlegend=True,
    )
    return fig

def calculate_month_difference(start_date_column, end_date_column, row):
    start_date = row[start_date_column]
    end_date = row[end_date_column]
    if pd.isna(start_date) or pd.isna(end_date):
        return None
    return relativedelta(end_date, start_date).years * 12 + relativedelta(end_date, start_date).months

def dep_to_received_claims_cohorts(df: pd.DataFrame) -> pd.DataFrame:
    grouped_df = df.groupby(['dateDepart_EndOfMonth', 'dateReceived_EndOfMonth'], as_index=False).agg({'clmNum_unique': 'sum'}).sort_values(['dateDepart_EndOfMonth', 'dateReceived_EndOfMonth'])
    for col in ['dateDepart_EndOfMonth', 'dateReceived_EndOfMonth']:
        grouped_df[f"{col}_str"] = grouped_df[col].astype("str")
    grouped_df["clm_past_present"] = "past"
    return grouped_df

def _develop_future_cohorts(future_pol_df:pd.DataFrame
                                ,selected_development:pd.DataFrame
                                ,month_diff_column_name:str
                                ,method:str
                                ,start_date_eom:str
                                ,start_date_month:str
                                ,end_date_eom:str
                                ,metric_column_name:str
                                ,past_futurename:str
                                ,metric_column_name_output:str=None
                                ,additional_group_col:list=None):
        
        """ Develop the vector of future policies from app to dep"""

        metric_column_name_total = f'{metric_column_name}_total'
        metric_column_name_cum = f'{metric_column_name}_cum'
        metric_column_name_diff = f'{metric_column_name}_diff'
        if metric_column_name_output is None:
            metric_column_name_diff_future = f'{metric_column_name}_'
        else:
            metric_column_name_diff_future = metric_column_name_output
        #past_futurename = f'{metric_column_name}_past_future'

        future_pol_df.drop(month_diff_column_name,axis=1,inplace=True, errors='ignore')
        # for cartesian product
        if method == "year":
            future_pol_df['key'] = 1
            selected_development['key'] = 1

            if additional_group_col is None:
                group_by_cols = [start_date_eom]
            else:
                group_by_cols = [*additional_group_col,start_date_eom]
            # Merge on the temporary key to achieve the cartesian product
            cartesian_df = pd.merge(future_pol_df, selected_development, on='key').drop('key', axis=1)
            cartesian_df.rename(columns={metric_column_name: metric_column_name_total}, inplace=True)
            cartesian_df[metric_column_name_cum] = cartesian_df[metric_column_name_total] * cartesian_df['cumulative_probability']

            # Ensure the DataFrame is sorted by 'dateApp_EndOfMonth' and 'app_to_depart_month' to accurately calculate differences within each cohort
            cartesian_df = cartesian_df.sort_values(by=[*group_by_cols, month_diff_column_name])

            # Calculate the difference in 'idpol_unique_cum' within each 'dateApp_EndOfMonth' group
            cartesian_df[metric_column_name_diff] = cartesian_df.groupby(group_by_cols)[metric_column_name_cum].diff()

            # The first entry of each group will be NaN because there's no previous value to subtract from. 
            # You can fill this with the 'idpol_unique_cum' value itself, assuming the difference from 0 for the first entry.
            cartesian_df[metric_column_name_diff].fillna(cartesian_df[metric_column_name_cum], inplace=True)

            # Renaming for clarity based on your intention
            cartesian_df.rename(columns={metric_column_name_diff: metric_column_name_diff_future}, inplace=True)

        elif method == "month":
            #future_pol_estimates_prior_year[start_date_month]
            future_pol_df[start_date_month] = pd.to_datetime(future_pol_df[start_date_eom]).dt.month

            if additional_group_col is None:
                group_by_cols = [start_date_eom,start_date_month]
            else:
                group_by_cols = [*additional_group_col,start_date_eom,start_date_month]

            cartesian_df = pd.merge(future_pol_df, selected_development, on=start_date_month)
            cartesian_df.rename(columns={metric_column_name: metric_column_name_total}, inplace=True)
            cartesian_df[metric_column_name_cum] = cartesian_df[metric_column_name_total] * cartesian_df['cumulative_probability']

            # Ensure the DataFrame is sorted by 'dateApp_EndOfMonth' and 'app_to_depart_month' to accurately calculate differences within each cohort
            cartesian_df = cartesian_df.sort_values(by=[*group_by_cols, month_diff_column_name])

            # Calculate the difference in 'idpol_unique_cum' within each 'dateApp_EndOfMonth' group
            cartesian_df[metric_column_name_diff] = cartesian_df.groupby(group_by_cols)[metric_column_name_cum].diff()

            # The first entry of each group will be NaN because there's no previous value to subtract from. 
            # You can fill this with the 'idpol_unique_cum' value itself, assuming the difference from 0 for the first entry.
            cartesian_df[metric_column_name_diff].fillna(cartesian_df[metric_column_name_cum], inplace=True)

            # # Renaming for clarity based on your intention
            cartesian_df.rename(columns={metric_column_name_diff: metric_column_name_diff_future}, inplace=True)

        # recreate the end date eom
        cartesian_df[end_date_eom] = cartesian_df.apply(
            lambda row: pd.to_datetime(row[start_date_eom]) + pd.DateOffset(months=int(row[month_diff_column_name])),
            axis=1
        )

        cartesian_df[past_futurename] = "future"

        return cartesian_df


# frequency_from_file_df_ = get_frequency_per_dep(segment=segment,cutoff=cutoff_date_frequency)
#  past_future_pol_cohorts_ = get_policy_count_per_dep(segment=segment,cutoff = cutoff_date, cutoff_finance=cutoff_date_finance)

def analyze_claims_forecast(
    segment,
    cutoff_date,
    cutoff_date_finance,
    cutoff_date_frequency,
    policies_df,
    claims_df,
    past_future_pol_cohorts_,
    frequency_from_file_df_,
    lag
):
    claims_df_seg = claims_df[claims_df[SEGMENT]==segment]
    claims_df_seg = claims_df_seg.groupby([DEPARTURE_DAY,RECEIVED_DAY,DATE_DEPART_END_OF_MONTH,DATE_RECEIVED_END_OF_MONTH]).agg({CLAIM_COUNT:"sum"}).reset_index()

    claims_df_seg["depart_to_receive_month"] = claims_df_seg.apply(lambda row: calculate_month_difference(DEPARTURE_DAY, RECEIVED_DAY, row), axis=1)
    claims_df_seg["clmNum_unique"] = claims_df_seg[CLAIM_COUNT]
    claims_df_seg["dateDepart"] = pd.to_datetime(claims_df_seg["dateDepart"])
    claims_df_seg["dateReceived"] = pd.to_datetime(claims_df_seg["dateReceived"])
    claims_df_seg[DATE_DEPART_END_OF_MONTH] = pd.to_datetime(claims_df_seg[DATE_DEPART_END_OF_MONTH])
    claims_df_seg[DATE_RECEIVED_END_OF_MONTH] = pd.to_datetime(claims_df_seg[DATE_RECEIVED_END_OF_MONTH])



    past_future_pol_cohorts_["dateDepart_EndOfMonth"] = pd.to_datetime(past_future_pol_cohorts_["dateDepart_EndOfMonth"])
    frequency_from_file_df_["dateDepart_EndOfMonth"] = pd.to_datetime(frequency_from_file_df_["dateDepart_EndOfMonth"])

    past_future_pol_cohorts_df_with_freq = pd.merge(past_future_pol_cohorts_,frequency_from_file_df_[["dateDepart_EndOfMonth","final_weighted_freq"]],how="left",on="dateDepart_EndOfMonth")

    # get the claim count per cohort
    past_future_pol_cohorts_df_with_freq["clmNum_unique_"] = past_future_pol_cohorts_df_with_freq["idpol_unique_"] * past_future_pol_cohorts_df_with_freq["final_weighted_freq"]

    # st.subheader("past_future_pol_cohorts_df_with_freq")
    # st.dataframe(past_future_pol_cohorts_df_with_freq)

    # Step 4: find development pattern
    month_diff_column_name = "depart_to_receive_month"
    #method_dep_to_rec = "year"
    level = 0.95
    metric_column_name="clmNum_unique"
    metric_column_name_output = "clmNum_unique_"
    start_date_column = "dateDepart"
    end_date_column = "dateReceived"
    past_futurename = "clm_past_present"
    method_dep_to_rec = "month"
    df_copy = claims_df_seg.copy()
    training_df = claims_df_seg.copy()
    extraction_date = cutoff_date


    monthly_dev_patterns, selected_pattern,dynamic_cols,dynamic_cols_end = select_main_development_info(df_copy
                                                                ,start_date_column = start_date_column
                                                                ,end_date_column = end_date_column
                                                                ,month_diff_column_name=month_diff_column_name
                                                                ,metric_column_name=metric_column_name
                                                                ,level=level
                                                                ,method=method_dep_to_rec)

    fig_dev_patterns = plot_development_patterns(monthly_dev_patterns
                                        ,selected_pattern
                                        ,month_diff_column_name
                                        ,dynamic_cols
                                        ,end_date=extraction_date
                                        ,xaxis_title="Departure to Receive Month"
                                        ,name_agg_line='Overall Development (avg by claim volume)'
                                        ,method=method_dep_to_rec
                                        ,show = False)

    selected_development = _select_development_to_forecast(selected_pattern,month_diff_column_name,dynamic_cols,method_dep_to_rec)

    # Step 5: forecast future claims from dep to receive
    future_claims_cohort_df_ = _develop_future_cohorts(past_future_pol_cohorts_df_with_freq
                                ,selected_development
                                ,month_diff_column_name
                                ,method = method_dep_to_rec
                                ,start_date_eom = dynamic_cols["start_date_eom"]
                                ,start_date_month = dynamic_cols["start_date_month"]
                                ,end_date_eom = dynamic_cols_end["end_date_eom"]
                                ,metric_column_name=metric_column_name_output
                                ,past_futurename = past_futurename
                                ,metric_column_name_output=metric_column_name_output
                                ,additional_group_col=None)

    # st.subheader("future_claims_cohort_df_")
    # st.dataframe(future_claims_cohort_df_)
    # Step 6: get clean past cohorts:
    past_claims_cohort_df = dep_to_received_claims_cohorts(training_df[training_df["dateReceived"]<=extraction_date])

    # fig_plot_dep_rec_twolines_past_cohorts = plot_dep_to_received_claims_twolines_cohorts(past_claims_cohort_df)
    past_claims_cohort_df["dateDepart_EndOfMonth"] = pd.to_datetime(past_claims_cohort_df["dateDepart_EndOfMonth"])
    past_claims_cohort_df["dateReceived_EndOfMonth"] = pd.to_datetime(past_claims_cohort_df["dateReceived_EndOfMonth"])

    future_claims_cohort_df_["dateDepart_EndOfMonth"] = pd.to_datetime(future_claims_cohort_df_["dateDepart_EndOfMonth"])
    future_claims_cohort_df_["dateReceived_EndOfMonth"] = pd.to_datetime(future_claims_cohort_df_["dateReceived_EndOfMonth"])

    # Step 6: get final cohorts to compute non reported yet claims
    final_claims_cohorts_ = future_claims_cohort_df_[["dateDepart_EndOfMonth","dateReceived_EndOfMonth","clmNum_unique_"]].merge(past_claims_cohort_df[["dateDepart_EndOfMonth","dateReceived_EndOfMonth","clmNum_unique"]]
                                                                                                                            ,how="outer"
                                                                                                                            ,on=["dateDepart_EndOfMonth","dateReceived_EndOfMonth"])

    if lag > 0:
        end_valid_date = pd.to_datetime(extraction_date) + pd.offsets.MonthEnd(-lag)
    else:
        end_valid_date = pd.to_datetime(extraction_date)

    # Corner Case #1: past cohorts with future received claims
    nyi_condition_1 = (final_claims_cohorts_["clmNum_unique"].isna())
    nyi_condition_2 = (final_claims_cohorts_["dateDepart_EndOfMonth"]<=extraction_date)
    nyi_condition_3 = (final_claims_cohorts_["dateReceived_EndOfMonth"]>extraction_date)
    #nyi_condition_2 = (final_claims_cohorts_["dateDepart_EndOfMonth"]<=end_valid_date)
    #nyi_condition_3 = (final_claims_cohorts_["dateReceived_EndOfMonth"]>end_valid_date)
    not_yet_incurred = final_claims_cohorts_[(nyi_condition_1) & (nyi_condition_2) & (nyi_condition_3)]


    past_claims_cohort_df_with_not_yet_incurred = pd.concat([past_claims_cohort_df[["dateDepart_EndOfMonth","dateReceived_EndOfMonth","clmNum_unique"]],
                                                            not_yet_incurred[["dateDepart_EndOfMonth","dateReceived_EndOfMonth","clmNum_unique_"]]],axis=0)

    to_develop_not_fully_dev_claims_cohorts = final_claims_cohorts_.groupby(["dateDepart_EndOfMonth"],as_index=False).agg({"clmNum_unique_":"sum"})
    past_claims_cohort_df_with_not_yet_incurred = pd.merge(past_claims_cohort_df_with_not_yet_incurred,to_develop_not_fully_dev_claims_cohorts,how="left",on="dateDepart_EndOfMonth",suffixes=["","target"])

    # Calculate the remaining amount needed for each 'dateDepart_EndOfMonth' to reach the target
    past_claims_cohort_df_with_not_yet_incurred['current_sum'] = past_claims_cohort_df_with_not_yet_incurred.groupby('dateDepart_EndOfMonth')['clmNum_unique'].transform('sum')
    past_claims_cohort_df_with_not_yet_incurred['expected_sum'] = past_claims_cohort_df_with_not_yet_incurred.groupby('dateDepart_EndOfMonth')['clmNum_unique_'].transform('sum')

    past_claims_cohort_df_with_not_yet_incurred['remaining_to_target'] = np.where(past_claims_cohort_df_with_not_yet_incurred['clmNum_unique_target'] >= past_claims_cohort_df_with_not_yet_incurred['current_sum'],
                                                                                    past_claims_cohort_df_with_not_yet_incurred['clmNum_unique_target'] - past_claims_cohort_df_with_not_yet_incurred['current_sum'],
                                                                                    0)

    # past_claims_cohort_df_with_not_yet_incurred['clmNum_unique_'] = np.where(
    #                                                                 past_claims_cohort_df_with_not_yet_incurred['remaining_to_target'] >= 0,
    #                                                                 past_claims_cohort_df_with_not_yet_incurred['clmNum_unique_'] * past_claims_cohort_df_with_not_yet_incurred['remaining_to_target'] / past_claims_cohort_df_with_not_yet_incurred['expected_sum'],
    #                                                                 0
    #                                                                 )
    past_claims_cohort_df_with_not_yet_incurred['clmNum_unique_'] = past_claims_cohort_df_with_not_yet_incurred['clmNum_unique_'] * past_claims_cohort_df_with_not_yet_incurred['remaining_to_target'] / past_claims_cohort_df_with_not_yet_incurred['expected_sum']

    # I don't want to create negative claims
    #past_claims_cohort_df_with_not_yet_incurred.loc[past_claims_cohort_df_with_not_yet_incurred['remaining_to_target'] < 0, 'clmNum_unique_'] = 0

    # Fill NaN in 'clmNum_unique_' with values from 'clmNum_unique'
    past_claims_cohort_df_with_not_yet_incurred["clmNum_unique_"] = past_claims_cohort_df_with_not_yet_incurred["clmNum_unique_"].fillna(past_claims_cohort_df_with_not_yet_incurred["clmNum_unique"])

    past_claims_cohort_df_with_not_yet_incurred["clm_past_present"] = "past"

    # CORNER CASE #2
    bttf_condition_1 = ~(final_claims_cohorts_["clmNum_unique"].isna())
    bttf_condition_2 = ~(final_claims_cohorts_["clmNum_unique_"].isna())
    #bttf_condition_3 = (final_claims_cohorts_["dateDepart_EndOfMonth"]>extraction_date)
    #btff_condition_4 = (final_claims_cohorts_["dateReceived_EndOfMonth"]<=extraction_date)
    bttf_condition_3 = (final_claims_cohorts_["dateDepart_EndOfMonth"]>end_valid_date)
    btff_condition_4 = (final_claims_cohorts_["dateReceived_EndOfMonth"]<=end_valid_date)

    not_yet_departed_already_incurred = final_claims_cohorts_[(bttf_condition_1) &(bttf_condition_3) & (btff_condition_4)]

    not_yet_departed_not_yet_incurred = final_claims_cohorts_[(bttf_condition_3) & (~btff_condition_4) & (bttf_condition_2)]

    future_claims_cohorts_with_not_yet_incurred = pd.concat([not_yet_departed_already_incurred[["dateDepart_EndOfMonth","dateReceived_EndOfMonth","clmNum_unique"]],
                                                            not_yet_departed_not_yet_incurred[["dateDepart_EndOfMonth","dateReceived_EndOfMonth","clmNum_unique_"]]],axis=0)

    future_claims_cohorts_with_not_yet_incurred_target = pd.merge(future_claims_cohorts_with_not_yet_incurred,to_develop_not_fully_dev_claims_cohorts,how="left",on="dateDepart_EndOfMonth",suffixes=["","target"])
    future_claims_cohorts_with_not_yet_incurred_target['current_sum'] = future_claims_cohorts_with_not_yet_incurred_target.groupby('dateDepart_EndOfMonth')['clmNum_unique'].transform('sum')
    future_claims_cohorts_with_not_yet_incurred_target['expected_sum'] = future_claims_cohorts_with_not_yet_incurred_target.groupby('dateDepart_EndOfMonth')['clmNum_unique_'].transform('sum')
    # don't create negative claims to reach the target
    future_claims_cohorts_with_not_yet_incurred_target['remaining_to_target'] = np.where(future_claims_cohorts_with_not_yet_incurred_target['clmNum_unique_target'] >= future_claims_cohorts_with_not_yet_incurred_target['current_sum'],
                                                                                    future_claims_cohorts_with_not_yet_incurred_target['clmNum_unique_target'] - future_claims_cohorts_with_not_yet_incurred_target['current_sum'],
                                                                                    0)
    future_claims_cohorts_with_not_yet_incurred_target['clmNum_unique_'] = future_claims_cohorts_with_not_yet_incurred_target['clmNum_unique_'] * future_claims_cohorts_with_not_yet_incurred_target['remaining_to_target'] / future_claims_cohorts_with_not_yet_incurred_target['expected_sum']

    future_claims_cohorts_with_not_yet_incurred_target["clmNum_unique_"] = future_claims_cohorts_with_not_yet_incurred_target["clmNum_unique_"].fillna(future_claims_cohorts_with_not_yet_incurred_target["clmNum_unique"])


    future_claims_cohorts_with_not_yet_incurred_target["clm_past_present"] = "future"
    # Use numpy.where to efficiently check each date and assign "future" or "past"
    #future_claims_cohorts_with_not_yet_incurred_target["clm_past_present"] = np.where(future_claims_cohorts_with_not_yet_incurred_target["dateDepart_EndOfMonth"] > extraction_date, "future", "past")

    # past_claims_cohort_df_with_not_yet_incurred["clm_past_present"] = "past"

    # Step 7: remove not needed cohorts from future
    #future_claims_cohorts_with_not_yet_incurred_target = future_claims_cohorts_with_not_yet_incurred_target[future_claims_cohorts_with_not_yet_incurred_target["dateReceived_EndOfMonth"]>extraction_date]

    # keep the rows where we did not already observed a received claim
    future_claims_cohorts_with_not_yet_incurred_target = future_claims_cohorts_with_not_yet_incurred_target[(future_claims_cohorts_with_not_yet_incurred_target["clmNum_unique"].isna())]

    # Step 8: Concat the end result
    past_future_claims_cohort_df_ = pd.concat([past_claims_cohort_df_with_not_yet_incurred[[past_futurename,"dateDepart_EndOfMonth","dateReceived_EndOfMonth",metric_column_name_output]]
                                        ,future_claims_cohorts_with_not_yet_incurred_target[[past_futurename,"dateDepart_EndOfMonth","dateReceived_EndOfMonth",metric_column_name_output]]])

    # if there's a lag, we need to remove the claims that are duplicated
    # we want to keep the expected claims coming from "future"
    # Find duplicated 'dateReceived_EndOfMonth' values (keeping all occurrences for further inspection)
    duplicated = past_future_claims_cohort_df_.duplicated(subset=["dateDepart_EndOfMonth",'dateReceived_EndOfMonth'], keep=False)

    # Filter for rows that are duplicated and have 'clm_past_present' marked as 'past'
    rows_to_drop = past_future_claims_cohort_df_[duplicated & (past_future_claims_cohort_df_['clm_past_present'] == 'past')]

    # Drop these rows from the DataFrame
    past_future_claims_cohort_df_ = past_future_claims_cohort_df_.drop(rows_to_drop.index)

    # Use numpy.where to efficiently check each date and assign "future" or "past"
    past_future_claims_cohort_df_["clm_past_present"] = np.where(past_future_claims_cohort_df_["dateDepart_EndOfMonth"] > extraction_date, "future", "past")

    #past_future_claims_cohort_df_ = past_future_claims_cohort_df_[past_future_claims_cohort_df_["dateReceived_EndOfMonth"]>extraction_date]
    # Corner Case #2: Future Cohort with received claims in the past:

    # remove the estimated received before the end_date_valuation
    # if config_file.get("lag",0) > 0:
    #     end_valid_date = pd.to_datetime(extraction_date) + pd.offsets.MonthEnd(-config_file.get("lag",0))
    # else:
    #     end_valid_date = pd.to_datetime(extraction_date)

    # past_future_claims_cohort_df_.loc[(past_future_claims_cohort_df_["dateReceived_EndOfMonth"]<=end_valid_date) & (past_future_claims_cohort_df_[past_futurename]=="future"),metric_column_name_output] = 0


    for col in ["dateDepart_EndOfMonth","dateReceived_EndOfMonth"]:
        past_future_claims_cohort_df_[f"{col}_str"] = past_future_claims_cohort_df_[col].astype("str")

    for col in ["dateDepart","dateReceived"]:
        past_future_claims_cohort_df_[col + '_year'] = pd.to_datetime(past_future_claims_cohort_df_[col + '_EndOfMonth']).dt.year
        past_future_claims_cohort_df_[col + '_month'] = pd.to_datetime(past_future_claims_cohort_df_[col + '_EndOfMonth']).dt.month


    start_x_axis= max(pd.to_datetime("2021-01-01"),past_future_claims_cohort_df_[DATE_RECEIVED_END_OF_MONTH].min())
    end_x_axis = min(pd.to_datetime("2027-12-31"),past_future_claims_cohort_df_[DATE_RECEIVED_END_OF_MONTH].max())

    final_claims_rec_data = past_future_claims_cohort_df_[past_future_claims_cohort_df_[DATE_RECEIVED_END_OF_MONTH]>=start_x_axis].groupby([DATE_RECEIVED_END_OF_MONTH],as_index=False).agg({"clmNum_unique_":"sum"})
    final_claims_rec_data[SEGMENT] = segment
    final_claims_rec_data["cutoff"] = pd.to_datetime(cutoff_date)
    final_claims_rec_data["cutoff_finance"] = pd.to_datetime(cutoff_date_finance)
    final_claims_rec_data["cutoff_frequency"] = pd.to_datetime(cutoff_date_frequency)
    final_claims_rec_data = final_claims_rec_data[["cutoff","cutoff_finance","cutoff_frequency",SEGMENT,DATE_RECEIVED_END_OF_MONTH,"clmNum_unique_"]]

    past_future_claims_cohort_df_[SEGMENT] = segment
    past_future_claims_cohort_df_["cutoff"] = pd.to_datetime(cutoff_date)
    past_future_claims_cohort_df_["cutoff_finance"] = pd.to_datetime(cutoff_date_finance)
    past_future_claims_cohort_df_["cutoff_frequency"] = pd.to_datetime(cutoff_date_frequency)
    
    past_future_claims_cohort_df_for_semantic_model = past_future_claims_cohort_df_[["cutoff","cutoff_finance","cutoff_frequency",SEGMENT,"clm_past_present",DATE_DEPART_END_OF_MONTH,DATE_RECEIVED_END_OF_MONTH,"clmNum_unique_"]]

    start_x_axis= max(pd.to_datetime("2021-01-01"),past_future_claims_cohort_df_[DATE_RECEIVED_END_OF_MONTH].min())
    final_claims_dep_data = past_future_claims_cohort_df_[past_future_claims_cohort_df_[DATE_DEPART_END_OF_MONTH]>=start_x_axis].groupby([DATE_DEPART_END_OF_MONTH],as_index=False).agg({"clmNum_unique_":"sum"})
    final_claims_dep_data[SEGMENT] = segment
    final_claims_dep_data["cutoff"] = pd.to_datetime(cutoff_date)
    final_claims_dep_data["cutoff_finance"] = pd.to_datetime(cutoff_date_finance)
    final_claims_dep_data["cutoff_frequency"] = pd.to_datetime(cutoff_date_frequency)

    final_claims_dep_data = final_claims_dep_data[["cutoff","cutoff_finance","cutoff_frequency",SEGMENT,DATE_DEPART_END_OF_MONTH,"clmNum_unique_"]]
    
    return {
        "fig_dev_patterns":fig_dev_patterns,
        "past_future_claims_cohort_df_":past_future_claims_cohort_df_,
        "final_claims_rec_data":final_claims_rec_data,
        "past_future_claims_cohort_df_for_semantic_model":past_future_claims_cohort_df_for_semantic_model,
        "final_claims_dep_data":final_claims_dep_data,
        "claims_df_seg":claims_df_seg
    }

#final_claims_rec_data, final_claims_dep_data, past_future_claims_cohort_df_for_semantic_model 



# def get_data():
#     pass

# def get_frequency_per_dep():
#     pass

# def get_policy_count_per_dep():
#     past_future_pol_cohorts[['pol_past_present','dateDepart_EndOfMonth','dateDepart_year','dateDepart_month','idpol_unique_']]
#     pass





# # load the frequency frequency_from_file_df_
# frequency_from_file_df_ = get_frequency_per_dep()

# # load the policy count per departure date
# past_future_pol_cohorts_ = get_policy_count_per_dep()

# past_future_pol_cohorts_df_with_freq = pd.merge(past_future_pol_cohorts_,frequency_from_file_df_[["dateDepart_EndOfMonth","final_weighted_freq"]],how="left",on="dateDepart_EndOfMonth")

# # get the claim count per cohort
# past_future_pol_cohorts_df_with_freq["clmNum_unique_"] = past_future_pol_cohorts_df_with_freq["idpol_unique_"] * past_future_pol_cohorts_df_with_freq["final_weighted_freq"]

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
# df_copy = training_df.copy()

# cutoff_date = '2025-04-30'
# segment = "Airbnb"
# lag=0
# extraction_date = cutoff_date



# monthly_dev_patterns, selected_pattern,dynamic_cols,dynamic_cols_end = select_main_development_info(df_copy
#                                                             ,start_date_column = start_date_column
#                                                             ,end_date_column = end_date_column
#                                                             ,month_diff_column_name=month_diff_column_name
#                                                             ,metric_column_name=metric_column_name
#                                                             ,level=level
#                                                             ,method=method_dep_to_rec)
# # fig_dev_patterns = plot_development_patterns(monthly_dev_patterns
# #                                     ,selected_pattern
# #                                     ,month_diff_column_name
# #                                     ,dynamic_cols
# #                                     ,end_date=extraction_date
# #                                     ,xaxis_title="Departure to Receive Month"
# #                                     ,name_agg_line='Overall Development (avg by claim volume)'
# #                                     ,method=method_dep_to_rec
# #                                     ,show = False)

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

# # Step 6: get clean past cohorts:
# past_claims_cohort_df = dep_to_received_claims_cohorts(training_df[training_df["dateReceived"]<=extraction_date])

# # fig_plot_dep_rec_twolines_past_cohorts = plot_dep_to_received_claims_twolines_cohorts(past_claims_cohort_df)

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

# fig_final_claims_dep_date = px.line(past_future_claims_cohort_df_.groupby("dateDepart_EndOfMonth",as_index=False).agg({"clmNum_unique_":"sum"}),x="dateDepart_EndOfMonth",y="clmNum_unique_")

# # r_plot_yoy_growth = plot_yoy_growth(past_future_pol_cohorts,metric_column="idpol_unique_",metric_name="Policy Volume",past_present_column="pol_past_present",start_year=2020)
# # r_fig_yoy_clm = plot_yoy_growth(past_future_claims_cohort_df_,metric_column="clmNum_unique_",metric_name="Claim Volume",past_present_column="clm_past_present",start_year=2020)
