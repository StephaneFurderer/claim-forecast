import pandas as pd
import os
import plotly.graph_objects as go
from pandas.tseries.offsets import MonthEnd, MonthBegin
from frequency_development import preprocess_data
# the goal is to create a forecat of future policy_count based on available information from (1) finance (2) extracted data
    
# get data for Airbnb
ROOT = "C:\\Users\\sfurderer\\OneLake - Microsoft\\USTI-ACTUARIAL-DEV\\USTI_IDEA_SILVER.Lakehouse\\Tables\\analysis\\"
ROOT_OUTPUT = "C:\\Users\\sfurderer\\OneLake - Microsoft\\USTI-ACTUARIAL-DEV\\USTI_IDEA_SILVER.Lakehouse\\Files\\frequency_forecast\\" 
CUTOFF_DATE = "2025-04-30"
from frequency_development import constants as const
# Use constants from the package
SEGMENT = const.SEGMENT
POLICY_COUNT = const.POLICY_COUNT
CLAIM_COUNT = const.CLAIM_COUNT
FREQUENCY_VAL = const.FREQUENCY_VAL
DEPARTURE_DAY = const.DEPARTURE_DAY
DEPARTURE_YEAR = const.DEPARTURE_YEAR
DEPARTURE_MONTH = const.DEPARTURE_MONTH
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


GCP = "gcp"
GCP_PER_POL = "gcp_per_pol"
POL_PAST_PRESENT = 'pol_past_present'

import frequency_development as fd
from policy_count_forecast.plot_utils import *
from dateutil.relativedelta import relativedelta

def calculate_month_difference(start_date_column, end_date_column, row):
    """Compute the month difference between two dates in a row."""
    start_date = row[start_date_column]
    end_date = row[end_date_column]
    if pd.isna(start_date) or pd.isna(end_date):
        return None
    return relativedelta(end_date, start_date).years * 12 + relativedelta(end_date, start_date).months




def load_data():
    
    cutoff_date_ = CUTOFF_DATE.replace("-","_")
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


def app_to_dep_policy_cohorts(df:pd.DataFrame)->pd.DataFrame:
    """ return the policy volume per purchase and departure dates from the training dataframe"""
    grouped_df = df.groupby([DATE_SOLD_END_OF_MONTH,DATE_DEPART_END_OF_MONTH],as_index=False).agg({POLICY_COUNT:"sum"}).sort_values([DATE_SOLD_END_OF_MONTH,DATE_DEPART_END_OF_MONTH])

    # represent the past cohorts
    grouped_df[POL_PAST_PRESENT] = "past"
    return grouped_df



def find_development_range(df:pd.DataFrame
                           , month_diff_column_name:str
                           , level:float=0.9
                           , *args
                           , **kwargs)->pd.DataFrame:
    """ find the lower and upper bound of the development """
    cut = (1.0-level)/2
    lower_bound = df[month_diff_column_name].quantile(cut)
    upper_bound = df[month_diff_column_name].quantile((level+cut))
    #print("{},{}".format(lower_bound,upper_bound))
    return lower_bound,upper_bound

def _prepare_data(df:pd.DataFrame
                  , lower_bound:float
                  , upper_bound:float
                  , month_diff_column_name:str
                  , metric_column_name: str
                  , dynamic_cols:dict
                  , dynamic_cols_end:dict
                  , agg_func:str = "sum"
                  , *args
                  , **kwargs)->pd.DataFrame:
    """ prepare data """
    # add development
    # df['departure_to_received_months'] = df.apply(calculate_month_difference, axis=1)
    # Assuming `lower_bound` and `upper_bound` have been calculated as before
    filtered_df = df[(df[month_diff_column_name] <= upper_bound) & (df[month_diff_column_name] >= lower_bound)]
    # First, sort the DataFrame as needed
    agg_list = list(dynamic_cols.values())
    end_date_eom = dynamic_cols_end['end_date_eom']

    filtered_df = filtered_df.sort_values(by=[*agg_list, end_date_eom])

    filtered_df = filtered_df.groupby([*agg_list, end_date_eom],as_index=False).agg({metric_column_name:agg_func})

    filtered_df[month_diff_column_name] = filtered_df.apply(lambda row: calculate_month_difference(dynamic_cols["start_date_eom"], end_date_eom, row), axis=1) 

    return filtered_df

def get_development_patterns(prepared_date:pd.DataFrame
                             , month_diff_column_name:str
                             , metric_column_name: str
                             , dynamic_cols:dict
                             , dynamic_cols_end:dict):
    # Perform aggregation and calculate cumsum within each group
    agg_list = list(dynamic_cols.values())
    end_date_eom = dynamic_cols_end['end_date_eom']

    aggregated_df = prepared_date.groupby(agg_list, as_index=False).apply(
        lambda x: x.assign(
            cumulative_sum=x[metric_column_name].cumsum(),
        )
    ).reset_index(drop=True)

    # Calculate the total sum for each group to normalize cumulative sums into probabilities
    aggregated_df['group_total'] = aggregated_df.groupby(agg_list)[metric_column_name].transform('sum')

    # Calculate cumulative probability
    aggregated_df['cumulative_probability'] = aggregated_df['cumulative_sum'] / aggregated_df['group_total']

    # Optionally, you might want to sort again if the apply operation changed the order
    aggregated_df = aggregated_df.sort_values(by=[*agg_list, month_diff_column_name])

    return aggregated_df


def get_selected_pattern(development_patterns:pd.DataFrame
                         , month_diff_column_name:str
                         , metric_column_name:str
                         , dynamic_cols:dict
                         , agg_func:str = "sum"
                         , method:str = "year")->pd.DataFrame:
    
    # we return only one development factor
    if method == "year":
        # Calculate the overall cumulative probability
        overall_cumulative_df = development_patterns.groupby(month_diff_column_name).agg({metric_column_name: agg_func}).reset_index()

        # Calculate the cumulative sum of 'clmNum_unique'
        overall_cumulative_df['cumulative_sum'] = overall_cumulative_df[metric_column_name].cumsum()

        # Calculate the total sum of 'clmNum_unique'
        total_sum_overall = overall_cumulative_df[metric_column_name].sum()

        # Calculate the cumulative probability
        overall_cumulative_df['cumulative_probability'] = overall_cumulative_df['cumulative_sum'] / total_sum_overall

        overall_cumulative_df['percentage_text'] = (overall_cumulative_df['cumulative_probability'] * 100).apply(lambda x: f'{x:.2f}%')

    elif method =="month":
        start_date_month = dynamic_cols['start_date_month']

        # Calculate the overall cumulative probability
        overall_cumulative_df = development_patterns.groupby([start_date_month,month_diff_column_name]).agg({metric_column_name: agg_func}).reset_index()

        # Compute cumulative sums and probabilities within each month group
        overall_cumulative_df['cumulative_sum'] = overall_cumulative_df.groupby(start_date_month)[metric_column_name].cumsum()
        
        # Compute the total sum for each month
        month_totals = overall_cumulative_df.groupby(start_date_month)['cumulative_sum'].transform('max')

        # Calculate the cumulative probability
        overall_cumulative_df['cumulative_probability'] = overall_cumulative_df['cumulative_sum'] / month_totals
        
        overall_cumulative_df['percentage_text'] = (overall_cumulative_df['cumulative_probability'] * 100).apply(lambda x: f'{x:.2f}%')


    return overall_cumulative_df

def select_main_development_info(df:pd.DataFrame
         , start_date_column: str
         , end_date_column: str
         , month_diff_column_name: str
         , metric_column_name: str
         , level:float=0.9
         , method:str = "year"):
    """Main function to calculate and plot development patterns between two dates."""

     # Compute additional columns
    df[start_date_column + '_EndOfMonth'] = pd.to_datetime(df[start_date_column]).dt.to_period('M').dt.to_timestamp() #pd.to_datetime(df[start_date_column]) + pd.offsets.MonthEnd(0)
    df[start_date_column + '_year'] = pd.to_datetime(df[start_date_column]).dt.year
    df[start_date_column + '_month'] = pd.to_datetime(df[start_date_column]).dt.month

    df[end_date_column + '_EndOfMonth'] = pd.to_datetime(df[end_date_column]).dt.to_period('M').dt.to_timestamp()  #pd.to_datetime(df[end_date_column]) + pd.offsets.MonthEnd(0)

    # Dictionary for additional dynamic column names to pass to other functions
    dynamic_cols = {
        'start_date_eom': start_date_column + '_EndOfMonth',
        'start_date_year': start_date_column + '_year',
        'start_date_month': start_date_column + '_month',
    }

    dynamic_cols_end = {
        'end_date_eom': end_date_column + '_EndOfMonth',
    }

    
    # too long to run >> Moved it to the load
    # df[month_diff_column_name] = df.apply(lambda row: calculate_month_difference(start_date_column, end_date_column, row), axis=1) 
    
    
    lower_bound, upper_bound = find_development_range(df
                                                      , month_diff_column_name
                                                      , level=level
                                                      )
    # find the development patterns for each departure month and year
    monthly_dev_patterns = get_development_patterns(_prepare_data(df
                                                                  , lower_bound
                                                                  , upper_bound
                                                                  , month_diff_column_name
                                                                  , metric_column_name
                                                                  , dynamic_cols
                                                                  , dynamic_cols_end)
                                                    , month_diff_column_name
                                                    , metric_column_name
                                                    , dynamic_cols
                                                    , dynamic_cols_end)
    # find the selected pattern (average of development patterns)
    selected_pattern = get_selected_pattern(monthly_dev_patterns
                                            , month_diff_column_name
                                            , metric_column_name
                                            , dynamic_cols
                                            , method=method
                                            )
    return monthly_dev_patterns,selected_pattern,dynamic_cols,dynamic_cols_end



def _select_development_to_forecast(selected_pattern_df:pd.DataFrame
                                    ,month_diff_column_name:str
                                    ,dynamic_cols:dict
                                    ,method:str)-> pd.DataFrame:

    start_date_month = dynamic_cols['start_date_month']
    
    if method == "year":
        selected_development = selected_pattern_df[[month_diff_column_name,"cumulative_probability"]]
    elif method == "month":
        selected_development = selected_pattern_df[[start_date_month,month_diff_column_name,"cumulative_probability"]]
    return selected_development




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
        lambda row: row[start_date_eom] + pd.DateOffset(months=int(row[month_diff_column_name])),
        axis=1
    )

    cartesian_df[past_futurename] = "future"

    return cartesian_df


def _merge_past_future_cohorts(past_cohorts_df:pd.DataFrame
                               ,future_cohorts_df:pd.DataFrame
                               ,start_date_eom:str
                               ,end_date_eom:str
                               ,month_diff_column_name:str
                               ,past_futurename:str
                               ,metric_column_name_output:str)->pd.DataFrame:

    past_future_pol_df_ = pd.concat([past_cohorts_df[[past_futurename,start_date_eom,end_date_eom,metric_column_name_output]]
                                     ,future_cohorts_df[[past_futurename,start_date_eom,end_date_eom,metric_column_name_output]]])

    past_future_pol_df_[month_diff_column_name] = past_future_pol_df_.apply(lambda row: calculate_month_difference(start_date_eom, end_date_eom, row), axis=1)

    for col in [start_date_eom,end_date_eom]:
        past_future_pol_df_[f"{col}_str"] = past_future_pol_df_[col].astype("str")

    return past_future_pol_df_



def main_pol_cohorts_development(df:pd.DataFrame
                                 ,future_pol:pd.DataFrame
                                 ,cohorts_past:pd.DataFrame
                                 ,start_date_column:str
                                 ,end_date_column:str
                                 ,month_diff_column_name:str
                                 ,metric_column_name:str
                                 ,past_futurename:str
                                 ,metric_column_name_output:str
                                 ,end_date:str
                                 ,level:float=0.975
                                 ,method:str="year"
                                 ,show:bool=False
                                 ):
    """ develop the cohort policies """
    monthly_dev_patterns, selected_pattern,dynamic_cols,dynamic_cols_end = select_main_development_info(df
                                                          ,start_date_column = start_date_column
                                                          ,end_date_column = end_date_column
                                                          ,month_diff_column_name=month_diff_column_name
                                                          ,metric_column_name=metric_column_name
                                                          ,level=level
                                                          ,method=method)
    fig_development = plot_development_patterns(monthly_dev_patterns
                                      , selected_pattern
                                      , month_diff_column_name
                                      , dynamic_cols
                                      , end_date = end_date
                                      , xaxis_title="App to Departure Month"
                                      , name_agg_line='Overall Development (avg by policy volume)'
                                      , method=method
                                      , show=show)
    
    selected_development = _select_development_to_forecast(selected_pattern,month_diff_column_name,dynamic_cols,method)

    cohorts_df_future = _develop_future_cohorts(future_pol
                                                           ,selected_development
                                                           ,month_diff_column_name
                                                           ,method
                                                           ,dynamic_cols['start_date_eom']
                                                           ,dynamic_cols['start_date_month']
                                                           ,dynamic_cols_end['end_date_eom']
                                                           ,metric_column_name=metric_column_name
                                                           ,past_futurename=past_futurename
                                                           ,metric_column_name_output=metric_column_name_output)
    
    cohorts_past[metric_column_name_output] = cohorts_past[metric_column_name]

    past_future_pol_cohorts_df_ = _merge_past_future_cohorts(cohorts_past
                                                             ,cohorts_df_future
                                                             ,dynamic_cols["start_date_eom"]
                                                             ,dynamic_cols_end["end_date_eom"]
                                                             ,month_diff_column_name
                                                             ,past_futurename
                                                             ,metric_column_name_output)
    
    past_future_pol_cohorts_df_["dateDepart_year"] = past_future_pol_cohorts_df_["dateDepart_EndOfMonth"].dt.year
    past_future_pol_cohorts_df_["dateDepart_month"] = past_future_pol_cohorts_df_["dateDepart_EndOfMonth"].dt.month

    past_future_pol_cohorts_df_["dateApp_year"] = past_future_pol_cohorts_df_["dateApp_EndOfMonth"].dt.year
    past_future_pol_cohorts_df_["dateApp_month"] = past_future_pol_cohorts_df_["dateApp_EndOfMonth"].dt.month

    return {"data":past_future_pol_cohorts_df_,"fig":fig_development} 


def get_training_cohorts(training_df):
    """ return policy training cohorts """
    cohorts_app_to_dep_pol = app_to_dep_policy_cohorts(training_df)
    fig = plot_app_to_dep_policy_cohorts(cohorts_app_to_dep_pol)
    return cohorts_app_to_dep_pol,fig


def _prepare_date(df:pd.DataFrame)-> pd.DataFrame:
    """
    Prepares the dataset by calculating cumulative policies, group totals, adjusted group totals,
    and probabilities for each month and year. Marks years as complete or incomplete.
    
    Parameters:
    - df (pd.DataFrame): The input DataFrame with policy data.
    
    Returns:
    - pd.DataFrame: The prepared DataFrame with additional calculations and markers for complete years.
    """

    # Aggregate data to get sum of idpol_unique per month per year
    monthly_policies = df.groupby(['dateDepart_EndOfMonth',DEPARTURE_YEAR, DEPARTURE_MONTH], as_index=False).agg({'idpol_unique': 'sum'}).sort_values([DEPARTURE_YEAR, DEPARTURE_MONTH])

    # Calculate cumulative policies per year
    monthly_policies['cumulative_policies'] = monthly_policies.groupby(DEPARTURE_YEAR)['idpol_unique'].cumsum()

    monthly_policies['group_total'] = monthly_policies.groupby([DEPARTURE_YEAR])['idpol_unique'].transform('sum')

    # Calculate the number of months present for each year
    months_present_per_year = monthly_policies.groupby(DEPARTURE_YEAR)[DEPARTURE_MONTH].nunique().reset_index()
    months_present_per_year.rename(columns={DEPARTURE_MONTH: 'months_present'}, inplace=True)
    # Merge the months_present back into the monthly_policies DataFrame
    monthly_policies = pd.merge(monthly_policies, months_present_per_year, on=DEPARTURE_YEAR)

    ### do we need those data at this stage?
    # Adjust 'group_total' for each year to account for the missing months
    monthly_policies['adjusted_group_total'] = monthly_policies['group_total'] * (12 / monthly_policies['months_present'])

    # Now, use 'adjusted_group_total' for further calculations
    # Calculate cumulative policies per year considering adjusted_group_total
    monthly_policies['cumulative_policies'] = monthly_policies.groupby(DEPARTURE_YEAR)['idpol_unique'].cumsum()

    # Calculate adjusted cumulative probability
    monthly_policies['adjusted_cumulative_probability'] = monthly_policies['cumulative_policies'] / monthly_policies['adjusted_group_total']

    # Calculate density probability with the adjusted group total
    monthly_policies['adjusted_density_probability'] = monthly_policies['idpol_unique'] / monthly_policies['adjusted_group_total']

    # Assuming 'months_present' is already calculated
    monthly_policies['is_complete_year'] = monthly_policies['months_present'] == 12

    return monthly_policies



def _get_a_posteriori_seasonality_on_complete_years(prepared_data:pd.DataFrame):
    """
    Calculates the weighted average frequency for complete years and normalizes the distribution.
    
    Parameters:
    - prepared_data (pd.DataFrame): The DataFrame prepared by _prepare_date function.
    
    Returns:
    - Tuple containing:
        - pd.DataFrame: Data for complete years with weighted frequency calculations.
        - pd.DataFrame: Normalized distribution for all months based on complete years.
    """

    # Calculate weighted frequencies using adjusted values
    # monthly_policies_complete = prepared_data[prepared_data["is_complete_year"]==True]

    prepared_data['weighted_freq'] = prepared_data['adjusted_density_probability'] * prepared_data['idpol_unique']
    weighted_avg = prepared_data.groupby(DEPARTURE_MONTH, as_index=False).agg({
        'weighted_freq': 'sum',
        'idpol_unique': 'sum'
    })

    # Compute final weighted average frequency using adjusted values
    weighted_avg['final_weighted_freq'] = weighted_avg['weighted_freq'] / weighted_avg['idpol_unique']

    # Fill missing months and calculate average as before
    all_months = pd.DataFrame({DEPARTURE_MONTH: range(1, 13)})
    weighted_avg[DEPARTURE_MONTH] = weighted_avg[DEPARTURE_MONTH].astype(int)

    # Ensure all months are present
    full_data = pd.merge(all_months, weighted_avg, on=DEPARTURE_MONTH, how='left')

    # Calculate the average of 'final_weighted_freq' excluding NaNs
    average_weighted_freq = full_data['final_weighted_freq'].mean()

    # Fill missing 'final_weighted_freq' values with the average
    full_data['final_weighted_freq'].fillna(average_weighted_freq, inplace=True)


    # Normalize 'final_weighted_freq' so that the sum equals 1
    full_data['final_weighted_freq'] /= full_data['final_weighted_freq'].sum()

    # After normalization, recheck that the sum equals 1
    #assert np.isclose(full_data['final_weighted_freq'].sum(), 1), "The sum of 'final_weighted_freq' does not equal 1."

    return prepared_data,full_data

def _realign_incomplete_data(prepared_data:pd.DataFrame,a_posteriori_data:pd.DataFrame):
    """
    Adjusts incomplete years based on the seasonality derived from complete years.
    
    Parameters:
    - prepared_data (pd.DataFrame): The DataFrame prepared by _prepare_date function.
    - a_posteriori_data (pd.DataFrame): Data containing the normalized distribution for all months.
    
    Returns:
    - pd.DataFrame: Adjusted data for incomplete years.
    """
    monthly_policies_incomplete = prepared_data[prepared_data["is_complete_year"] == False]

    # Merge the final_weighted_freq from full_data into monthly_policies_incomplete
    monthly_policies_incomplete = pd.merge(monthly_policies_incomplete, a_posteriori_data[[DEPARTURE_MONTH, 'final_weighted_freq']], on=DEPARTURE_MONTH, how='left')

    # Calculate the cumulative sum of final_weighted_freq for each incomplete year
    monthly_policies_incomplete['cumulative_final_weighted_freq'] = monthly_policies_incomplete.groupby(DEPARTURE_YEAR)['final_weighted_freq'].cumsum()

    # Adjust 'group_total' for each year to account for the missing months
    monthly_policies_incomplete['adjusted_group_total'] = monthly_policies_incomplete['group_total'] /  monthly_policies_incomplete.groupby([DEPARTURE_YEAR])['cumulative_final_weighted_freq'].transform('max') #* (12 / monthly_policies['months_present'])

    # Now, use 'adjusted_group_total' for further calculations
    # Calculate cumulative policies per year considering adjusted_group_total
    monthly_policies_incomplete['cumulative_policies'] = monthly_policies_incomplete.groupby(DEPARTURE_YEAR)['idpol_unique'].cumsum()

    # Calculate adjusted cumulative probability
    monthly_policies_incomplete['adjusted_cumulative_probability'] = monthly_policies_incomplete['cumulative_policies'] / monthly_policies_incomplete['adjusted_group_total']

    # Calculate density probability with the adjusted group total
    monthly_policies_incomplete['adjusted_density_probability'] = monthly_policies_incomplete['idpol_unique'] / monthly_policies_incomplete['adjusted_group_total']

    return monthly_policies_incomplete


def _build_all_years_distribution(complete_data,incomplete_data):
    """
    Merges adjusted distributions from complete and incomplete years.
    
    Parameters:
    - complete_data (pd.DataFrame): Adjusted data for complete years.
    - incomplete_data (pd.DataFrame): Adjusted data for incomplete years.
    
    Returns:
    - pd.DataFrame: Combined distribution for all years.
    """
    return pd.concat([complete_data, incomplete_data]).sort_values('dateDepart_EndOfMonth')

def _filter_df_to_exclude_dates(df,exclude_dates=None, exclude_ranges=None):
    # Convert exclude_dates to datetime year-month for easy comparison
    temp = df.copy()
    if exclude_dates:
        exclude_dates = [pd.to_datetime(date).strftime('%Y-%m') for date in exclude_dates]
    
    # Filter based on specific dates to exclude
    if exclude_dates:
        temp = temp[~temp['dateDepart_EndOfMonth'].dt.strftime('%Y-%m').isin(exclude_dates)]
    
    # Filter based on multiple ranges of dates to exclude
    if exclude_ranges:
        for start_date, end_date in exclude_ranges:
            start_date = pd.to_datetime(start_date) + pd.offsets.MonthBegin(0)
            end_date = pd.to_datetime(end_date) + pd.offsets.MonthEnd(0)
            # print("{}:{}".format(start_date,end_date))
            temp = temp[~((temp['dateDepart_EndOfMonth'] >= start_date) & (temp['dateDepart_EndOfMonth'] <= end_date))]
    return temp


def compute_final_seasonality(all_years_distribution:pd.DataFrame,exclude_dates=None, exclude_ranges=None):
    """
    Applies date or range exclusions and recalculates the weighted average frequency across all years.
    
    Parameters:
    - all_years_distribution (pd.DataFrame): Combined distribution for all years.
    - exclude_dates (list): Specific dates to exclude.
    - exclude_ranges (list of tuples): Date ranges to exclude.
    
    Returns:
    - pd.DataFrame: The final seasonality data with adjustments and exclusions applied.
    """

    temp = _filter_df_to_exclude_dates(all_years_distribution, exclude_dates,exclude_ranges)
   
    # Calculate weighted frequencies using adjusted values
    temp['weighted_freq'] = temp['adjusted_density_probability'] * temp['idpol_unique']
    weighted_avg = temp.groupby(DEPARTURE_MONTH, as_index=False).agg({
        'weighted_freq': 'sum',
        'idpol_unique': 'sum'
    })

    # Compute final weighted average frequency using adjusted values
    weighted_avg['final_weighted_freq'] = weighted_avg['weighted_freq'] / weighted_avg['idpol_unique']

    # Fill missing months and calculate average as before
    all_months = pd.DataFrame({DEPARTURE_MONTH: range(1, 13)})
    weighted_avg[DEPARTURE_MONTH] = weighted_avg[DEPARTURE_MONTH].astype(int)

    # Ensure all months are present
    full_data = pd.merge(all_months, weighted_avg, on=DEPARTURE_MONTH, how='left')

    # Calculate the average of 'final_weighted_freq' excluding NaNs
    average_weighted_freq = full_data['final_weighted_freq'].mean()

    # Fill missing 'final_weighted_freq' values with the average
    full_data['final_weighted_freq'].fillna(average_weighted_freq, inplace=True)


    # Normalize 'final_weighted_freq' so that the sum equals 1
    full_data['final_weighted_freq'] /= full_data['final_weighted_freq'].sum()

    # After normalization, recheck that the sum equals 1
    #assert np.isclose(full_data['final_weighted_freq'].sum(), 1), "The sum of 'final_weighted_freq' does not equal 1."


    # Calculate percentage text for the adjusted final weighted frequency
    full_data['percentage_text'] = (full_data['final_weighted_freq'] * 100).apply(lambda x: f'{x:.2f}%')

    return full_data


def _get_exclude_dates_ranges(config_file,section:str="frequency"):
    # Default configuration in case the partner or parameters are missing
    default_config = {'exclude_dates': [], 'exclude_ranges': []}
    # Check if partner is in the configuration and has the expected keys
    config_freq_lob = config_file.get(section, default_config)

    # Extract 'exclude_dates', ensuring it defaults to an empty string if missing
    exclude_dates = config_freq_lob.get("exclude_dates", [])
    # Split the exclude_dates string into a list if it's not empty, else default to an empty list
    # Safely extract 'exclude_ranges' and transform each nested list into a tuple
    exclude_ranges_raw = config_freq_lob.get('exclude_ranges', [])
    exclude_ranges = [(date_range[0], date_range[1]) for date_range in exclude_ranges_raw if len(date_range) == 2]
    #print(f"{exclude_dates},{exclude_ranges}")
    return exclude_dates, exclude_ranges


def plot(all_years_distribution,selected_data,exclude_dates=None, exclude_ranges=None,):

    df = _filter_df_to_exclude_dates(all_years_distribution,exclude_dates=exclude_dates, exclude_ranges=exclude_ranges)

    grayscale_start = 0.8
    grayscale_end = 0.1
    unique_years = sorted(all_years_distribution[DEPARTURE_YEAR].unique())
    grayscale_colors = np.linspace(grayscale_start, grayscale_end, len(unique_years))

    # Create figure and add traces for each year in grayscale
    fig = go.Figure()
    for i, year in enumerate(unique_years):
        df_year = df[df[DEPARTURE_YEAR] == year]
        color = f'rgb({grayscale_colors[i]*255}, {grayscale_colors[i]*255}, {grayscale_colors[i]*255})'
        fig.add_trace(go.Scatter(x=df_year[DEPARTURE_MONTH], y=df_year['adjusted_density_probability'], mode='lines', name=str(year), line=dict(color=color)))

    # # Calculate percentage for the weighted average frequency and format it as text
    # df['percentage_text'] = (df['final_weighted_freq'] * 100).apply(lambda x: f'{x:.2f}%')

    # Add the weighted average frequency as an additional line in red with text labels
    fig.add_trace(go.Scatter(x=selected_data[DEPARTURE_MONTH], y=selected_data['final_weighted_freq'],
                            mode='lines+text', name='Weighted Average', line=dict(color='red', width=2),
                            text=selected_data['percentage_text'], textposition="top center"))

    # Update layout for clarity
    fig.update_layout(
        title='Monthly Policy Distribution by Year and Weighted Average',
        xaxis_title='Month',
        yaxis_title='Frequency (%)',
        yaxis=dict(tickformat=".2%"),
        xaxis=dict(tickmode='array', tickvals=list(range(1, 13)), ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']),
        plot_bgcolor="white"
    )

    return fig

def _forecast_pol(df
         , end_date:str
         , x:str=DATE_DEPART_END_OF_MONTH
         , forecast_end_date:str = '2027-12-31'
         , lag:int = 0
         , growth_rates = {}
         ):
    """
    Orchestrates the workflow to compute the final seasonality data.
    """

    # growth_rates= {
    #   "2023": 0,
    #   "2024": -0.16,
    #   "2025": 0.0369,
    #   "2026": -0.05,
    #   "2027": 0
    # }
    growth_rates_int_keys = {int(key): value for key, value in growth_rates.items()}

    forecast_end_date_ = pd.to_datetime(end_date) + relativedelta(years = 3)
    forecast_end_date = pd.Timestamp(year=forecast_end_date_.year,month=12,day=31)
    #exclude_dates, exclude_ranges = _get_exclude_dates_ranges(config_file,section="policy_forecast")

    #effective_lag = lag if lag is not None else config_file.get("lag", 0)

    end_valid_date = pd.to_datetime(end_date) + pd.offsets.MonthEnd(-lag) # pd.offsets.MonthEnd(-effective_lag) # pd.offsets.MonthEnd(-config_file.get("lag",0))

    df_input = df.copy()
    df_input = df_input[df_input[x]<=end_valid_date]

    prepared_data = _prepare_date(df_input)
    complete_data,a_posteriori_data = _get_a_posteriori_seasonality_on_complete_years(prepared_data)
    incomplete_data_realigned = _realign_incomplete_data(prepared_data,a_posteriori_data)
    all_years_data = _build_all_years_distribution(complete_data[complete_data["is_complete_year"]==True]
                                                   ,incomplete_data_realigned[incomplete_data_realigned["is_complete_year"]==False])
    print("exclude_ranges included <<<<<<<<<")
    selected_data = compute_final_seasonality(all_years_data,exclude_dates=None,exclude_ranges=[["2018-01","2020-12"]])

    fig_distribution_selection = plot(all_years_data,selected_data)


    #growth_rates = config_file.get("policy_forecast",{}).get("growth_rates",{})
    # Convert the keys from strings to integers
    
    # growth_rates_df = pd.DataFrame(list(growth_rates.items()), columns=['dateDepart_year', 'growth_rate'])
    # growth_rates_df["dateDepart_year"] = growth_rates_df["dateDepart_year"].astype(int)

    future_months = []  
    current_date = pd.to_datetime(end_valid_date) + MonthBegin(1)  # Example start date

    while current_date <= forecast_end_date:
        future_months.append(current_date)
        current_date = (current_date + MonthBegin(1))

    # Convert future_months to a DataFrame
    future_df = pd.DataFrame({'dateDepart_EndOfMonth': future_months})
    future_df[DEPARTURE_YEAR] = future_df['dateDepart_EndOfMonth'].dt.year
    future_df[DEPARTURE_MONTH] = future_df['dateDepart_EndOfMonth'].dt.month

    future_df = pd.merge(future_df, selected_data[[DEPARTURE_MONTH, 'final_weighted_freq']], on=DEPARTURE_MONTH, how='left')

    max_year = all_years_data[DEPARTURE_YEAR].max()
    max_month = all_years_data[all_years_data[DEPARTURE_YEAR]==max_year][DEPARTURE_MONTH].max()

    if max_month < 12: # still have month in the current year to forecast
        last_year = all_years_data[DEPARTURE_YEAR].max()-1
    elif max_month == 12: # no more month in the current year to forecast
        last_year = all_years_data[DEPARTURE_YEAR].max()
    else: #should raise an error
        raise ValueError(f"max_month value {max_month} is invalid. It should be within 1 to 12.")

    try:
    # Attempt to get the last value of 'adjusted_group_total' for the last year
        last_year_volume = all_years_data[all_years_data[DEPARTURE_YEAR] == last_year]['adjusted_group_total'].iloc[-1]
    except (IndexError, KeyError):
    # If an error occurs (e.g., no data for the last year or the column doesn't exist), set to 0
        last_year_volume = 1
    

    # Initialize a column for cumulative policies
    future_df['cumulative_policies'] = None

    # Calculate cumulative policies for each future year based on growth rates
    for year in future_df[DEPARTURE_YEAR].unique():
        if year in growth_rates_int_keys:
            last_year_volume *= (1 + growth_rates_int_keys[year])
        future_df.loc[future_df[DEPARTURE_YEAR] == year, 'cumulative_policies'] = last_year_volume


    future_df['monthly_forecasted_policies'] = future_df['cumulative_policies'] * future_df['final_weighted_freq']

        # Step 1: Create a flag for past and future data
    all_years_data['pol_past_present'] = 'past'
    future_df['pol_past_present'] = 'future'

    # Step 2: Harmonize column names
    # Assume 'monthly_forecasted_policies' in future_df corresponds to 'idpol_unique' in all_years_distribution
    # If 'monthly_forecasted_policies' does not exist yet, you might first need to calculate it in future_df
    # This example will proceed assuming it's ready to be harmonized

    # Ensure 'idpol_unique' exists in future_df for concatenation, even if it will be NaN or some placeholder values
    future_df['idpol_unique_'] = future_df['monthly_forecasted_policies']  # Or any other calculation that fits your use case

    # Now, let's rename columns to match between the two dataframes for a seamless concatenation
    # Note: Adjust column names as necessary to match your exact DataFrame structures
    all_years_data.rename(columns={'idpol_unique': 'idpol_unique_'}, inplace=True)

    # Step 3: Concatenate the DataFrames
    combined_df = pd.concat([all_years_data[['pol_past_present','dateDepart_EndOfMonth',DEPARTURE_YEAR,DEPARTURE_MONTH,'idpol_unique_','adjusted_cumulative_probability','adjusted_density_probability']], future_df], ignore_index=True)


    # Step 4: quick fix: recompute the idpol_unique_ because the step 3 did not work. << When time def redo this function but should work as is.

    # Filter the past and future data
    df = combined_df.copy()
    past_data = df[df['pol_past_present'] == 'past']
    future_data = df[df['pol_past_present'] == 'future']

    # Calculate the total observed policies for the past in each year
    past_policies_by_year = past_data.groupby(DEPARTURE_YEAR)['idpol_unique_'].sum()

    # Get the cumulative policies expected for the future years
    cumulative_policies = future_data.groupby(DEPARTURE_YEAR)['cumulative_policies'].first()

    # Calculate the difference to be allocated for future years
    difference_to_allocate = cumulative_policies.subtract(past_policies_by_year, fill_value=0).to_frame(name='policies_to_allocate').reset_index()

    # Merge the future data with the differences to allocate on 'dateDepart_year'
    future_data = future_data.merge(difference_to_allocate, on=DEPARTURE_YEAR, how='left')

    # Calculate the total weighted frequency for future data by year to normalize
    total_weighted_freq_by_year = future_data.groupby(DEPARTURE_YEAR)['final_weighted_freq'].transform('sum')

    # Allocate the differences based on the final_weighted_freq
    future_data['allocated_policies'] = (
        future_data['final_weighted_freq'] / total_weighted_freq_by_year * future_data['policies_to_allocate']
    )

    # Combine the past and future data, now including allocated policies for future
    df_combined = pd.concat([past_data, future_data])

    # Use 'idpol_unique_' from the past data where 'allocated_policies' is NaN
    df_combined['allocated_policies'] = df_combined['allocated_policies'].fillna(df_combined['idpol_unique_'])

    # Drop the 'idpol_unique_' column
    df_combined = df_combined.drop(columns=['idpol_unique_'])

    # Rename the 'allocated_policies' column to 'idpol_unique_'
    df_combined = df_combined.rename(columns={'allocated_policies': 'idpol_unique_'})

    return df_combined,fig_distribution_selection


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
            user_config = {"lag": 0}
        all_configs[config_key] = user_config
    return user_config

def load_data(cutoff_date,root:str):
    
    cutoff_date_ = cutoff_date.replace("-","_")
    claims_file = root + f"_clm_count_{cutoff_date_}"
    policies_file = root + f"_pol_count_{cutoff_date_}"
    claims_df = pd.read_parquet(claims_file)
    policies_df = pd.read_parquet(policies_file)

    for col in [SOLD_DAY,DEPARTURE_DAY,DATE_SOLD_END_OF_MONTH,DATE_DEPART_END_OF_MONTH]:
        policies_df[col] = pd.to_datetime(policies_df[col]).dt.tz_localize(None)
    
    for col in [SOLD_DAY,DEPARTURE_DAY,RECEIVED_DAY,DATE_SOLD_END_OF_MONTH,DATE_DEPART_END_OF_MONTH,DATE_RECEIVED_END_OF_MONTH]:
        claims_df[col] = pd.to_datetime(claims_df[col]).dt.tz_localize(None)
    
    policies_df , claims_df = preprocess_data(policies_df,claims_df)
    return policies_df , claims_df


def prepare_finance_df(df,metric:str,format_date=6): 
    date_column = 'RULOB'
    # Convert to string and pad with zeros to ensure YYYYMM format
    df[date_column] = df[date_column].astype(str).str.zfill(format_date) 

    # Fix malformed or missing data (e.g., "0000-na") before formatting
    df[date_column] = df[date_column].apply(lambda x: '0001-01' if 'na' in x else x[:4] + '-' + x[4:6])

    # Convert to datetime, handling errors with 'coerce' to NaT for any remaining problematic data
    df[date_column] = pd.to_datetime(df[date_column], format='%Y-%m', errors='coerce')

    df.dropna(inplace=True,axis=0,how='all')
    df.dropna(inplace=True,axis=1,how='all')
    # Unpivot the DataFrame
    #df.columns = [col.replace("VRBO", "HomeAway") for col in df.columns]
    replacement_dict = {
    'VRBO TI': 'HomeAway',
    'VRBO PDP': 'HomeAway-PDP',
    'Property Security Deposit':'Security Deposit',
    'In House(B2C)':'In House',
    'Supplier(MSC Cruises)':'ClubMed (hist Supplier)',
    'Expedia Travel Package':'Expedia'
    }
    #print(df.head())
    df = pd.melt(df, id_vars=['RULOB'], var_name=SEGMENT, value_name=metric)
    df[SEGMENT] = df[SEGMENT].replace(replacement_dict)
    df.rename({date_column:DATE_SOLD_END_OF_MONTH},axis=1,inplace=True)
    return df


def load_gcp_assumptions(date_str:str,finance_input_folder:str):
        source_filename = finance_input_folder + date_str + "\\gcp_finance.csv"

        gcp_finance_exists = os.path.exists(source_filename)

        # Check if result file exists and overwrite is False
        if (gcp_finance_exists):
            gcp_finance = pd.read_csv(source_filename,dtype = float)
            gcp_finance_prep = prepare_finance_df(gcp_finance,metric="gcp",format_date=6)

            # should raise error if find nothing!
            # gcp_finance_prep_partner = gcp_finance_prep[gcp_finance_prep[SEGMENT]==segment]
        return gcp_finance_prep
    
def load_pol_count_assumptions(date_str:str,finance_input_folder:str):
    source_filename = finance_input_folder + date_str + "\\pol_count_finance.csv"

    gcp_finance_exists = os.path.exists(source_filename)

    # Check if result file exists and overwrite is False
    if (gcp_finance_exists):
        gcp_finance = pd.read_csv(source_filename,dtype = float)
        gcp_finance_prep = prepare_finance_df(gcp_finance,metric="idpol_unique",format_date=6)

        # should raise error if find nothing!
        gcp_finance_prep_partner = gcp_finance_prep.rename(columns={"idpol_unique":POLICY_COUNT})
        return gcp_finance_prep_partner


def get_gcp_per_pol_from_finance(finance_cutoff_date,finance_input_folder):

    gcp_df = load_gcp_assumptions(date_str=finance_cutoff_date,finance_input_folder=finance_input_folder)
    pol_count_finance_df = load_pol_count_assumptions(date_str = finance_cutoff_date,finance_input_folder=finance_input_folder)
    gcp_per_pol = pd.merge(gcp_df
                            ,pol_count_finance_df
                            ,on=[DATE_SOLD_END_OF_MONTH,SEGMENT],how="left")
        
    gcp_per_pol[GCP_PER_POL] = np.where(gcp_per_pol[POLICY_COUNT].astype('float') <= 0 , np.nan , gcp_per_pol[GCP] / gcp_per_pol[POLICY_COUNT].astype('float'))

    gcp_per_pol['finance_cutoff_date'] = finance_cutoff_date
    return gcp_per_pol[['finance_cutoff_date',SEGMENT,DATE_SOLD_END_OF_MONTH,GCP,POLICY_COUNT,GCP_PER_POL]]


def forecast_policy_count(policies_seg:pd.DataFrame
                          , segment:str
                          , cutoff_date:str
                          , gcp_per_pol_seg:pd.DataFrame
                          , cutoff_date_finance:str):
    """
    
    """
    #gcp_per_pol = get_gcp_per_pol_from_finance(segment,cutoff_date_finance)
    condition_2 = (gcp_per_pol_seg[DATE_SOLD_END_OF_MONTH]>=cutoff_date)
    shocked_future_pol = gcp_per_pol_seg[condition_2]

    policies_seg["app_to_depart_month"] = policies_seg.apply(lambda row: calculate_month_difference(SOLD_DAY, DEPARTURE_DAY, row), axis=1)
    policies_seg["idpol_unique"] = policies_seg[POLICY_COUNT]
    policies_seg['dateDepart_year'] = policies_seg['dateDepart'].dt.year
    policies_seg['dateDepart_month'] = policies_seg['dateDepart'].dt.month

    cohorts_app_to_dep_pol,fig_training_cohorts = get_training_cohorts(training_df=policies_seg)


    month_diff_column_name = "app_to_depart_month"
    level = 0.85
    metric_column_name=POLICY_COUNT
    metric_column_name_output = "idpol_unique_"
    start_date_column = SOLD_DAY
    end_date_column = DEPARTURE_DAY
    past_futurename = POL_PAST_PRESENT
    method_app_to_dep = "month"

    r_future_pol_cohorts = main_pol_cohorts_development(df=policies_seg
                                ,end_date=cutoff_date
                                ,future_pol = shocked_future_pol
                                ,cohorts_past=cohorts_app_to_dep_pol
                                ,start_date_column=start_date_column
                                ,end_date_column=end_date_column
                                ,month_diff_column_name=month_diff_column_name
                                ,metric_column_name=metric_column_name
                                ,past_futurename = past_futurename
                                ,level=level
                                ,method = method_app_to_dep
                                ,show = False
                                ,metric_column_name_output=metric_column_name_output)

    past_future_pol_cohorts_df_ = r_future_pol_cohorts["data"]

    #st.plotly_chart(r_future_pol_cohorts["fig"],use_container_width=True, key="r_future_pol_cohorts_fig")

    new_past_future_pol_cohorts_df_ = past_future_pol_cohorts_df_.groupby(DATE_DEPART_END_OF_MONTH,as_index=False).agg({"idpol_unique_":"sum"})
    past_future_pol_cohorts = new_past_future_pol_cohorts_df_.copy()

    past_future_pol_cohorts[POL_PAST_PRESENT] = np.where(past_future_pol_cohorts[DATE_DEPART_END_OF_MONTH]>cutoff_date,"future","past")
    past_future_pol_cohorts[SEGMENT] = segment
    past_future_pol_cohorts["cutoff"] = cutoff_date
    past_future_pol_cohorts["cutoff_finance"] = cutoff_date_finance
    past_future_pol_cohorts = past_future_pol_cohorts[["cutoff","cutoff_finance",SEGMENT,POL_PAST_PRESENT,DATE_DEPART_END_OF_MONTH,"idpol_unique_"]]

    per_app_date_ = past_future_pol_cohorts_df_.groupby(DATE_SOLD_END_OF_MONTH,as_index=False).agg({"idpol_unique_":"sum"})
    per_app_date_[POL_PAST_PRESENT] = np.where(per_app_date_[DATE_SOLD_END_OF_MONTH]>cutoff_date,"future","past")
    per_app_date_[SEGMENT] = segment
    per_app_date_["cutoff"] = cutoff_date
    per_app_date_["cutoff_finance"] = cutoff_date_finance
    per_app_date_ = per_app_date_[["cutoff","cutoff_finance",SEGMENT,POL_PAST_PRESENT,DATE_SOLD_END_OF_MONTH,"idpol_unique_"]]

    return past_future_pol_cohorts,per_app_date_,past_future_pol_cohorts_df_


def save_data(segment, cutoff_date, cutoff_date_finance, past_future_pol_cohorts, per_app_date_, results_path="_results"):
    # Ensure results folder exists
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    cutoff_date = pd.to_datetime(cutoff_date)
    cutoff_str = cutoff_date

    # Prepare best frequencies DataFrame
    r1_df = past_future_pol_cohorts.copy()
    r2_df = per_app_date_.copy()
    
    # Save best frequencies
    r1 = os.path.join(results_path, 'pol_count_per_dep_.csv')
    r2 = os.path.join(results_path, 'pol_count_per_app_.csv')

    def _save(r,df):
        if os.path.exists(r):
            existing_r = pd.read_csv(r)
            mask = ~(
                (existing_r[SEGMENT] == segment) &
                (existing_r['cutoff'] == cutoff_str) & 
                (existing_r['cutoff_finance'] == cutoff_date_finance)
            )
            existing_r = existing_r[mask]
            existing_r = pd.concat([existing_r, df], ignore_index=True)
        else:
            existing_r = df
        existing_r.to_csv(r, index=False)

    _save(r1,r1_df)
    _save(r2,r2_df)

    # if os.path.exists(r1):
    #     existing_r1 = pd.read_csv(r1)
    #     mask = ~(
    #         (existing_r1[SEGMENT] == segment) &
    #         (existing_r1['cutoff'] == cutoff_str)
    #     )
    #     existing_r1 = existing_r1[mask]
    #     existing_r1 = pd.concat([existing_r1, r1_df], ignore_index=True)
    # else:
    #     existing_r1 = r1_df
    # existing_r1.to_csv(r1, index=False)

def save_data_tm(segment, cutoff_date, cutoff_date_finance,past_future_pol_cohorts, results_path="_results"):
    # Ensure results folder exists
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    cutoff_date = pd.to_datetime(cutoff_date)
    cutoff_str = cutoff_date

    # Prepare best frequencies DataFrame
    r1_df = past_future_pol_cohorts.copy()
    #print(r1_df.tail())
    # Save best frequencies
    r1 = os.path.join(results_path, 'pol_count_per_dep_.csv')
    #print(r1)
    def _save(r,df):
        if os.path.exists(r):
            existing_r = pd.read_csv(r).dropna()
            mask = ~(
                (existing_r[SEGMENT] == segment) &
                (existing_r['cutoff'] == cutoff_str) & 
                (existing_r['cutoff_finance'] == cutoff_date_finance)
            )
            existing_r = existing_r[mask]
            existing_r = pd.concat([existing_r, df], ignore_index=True)
        else:
            existing_r = df
        #print(existing_r.tail())
        existing_r.to_csv(r, index=False)

    _save(r1,r1_df)
   

if __name__ == "__main__":    

    segment = "Airbnb"
    block = "csa"

    policies_seg = pd.read_csv("../data/policies.csv")
    policies_seg[SOLD_DAY] = pd.to_datetime(policies_seg[SOLD_DAY])
    policies_seg[DEPARTURE_DAY] = pd.to_datetime(policies_seg[DEPARTURE_DAY])
    
    policies_seg["app_to_depart_month"] = policies_seg.apply(lambda row: calculate_month_difference(SOLD_DAY, DEPARTURE_DAY, row), axis=1)
    
    if block == "csa":

        finance_cutoff_date = '2025-03-31'
        
        # def get_gcp_per_pol_from_finance(segment,finance_cutoff_date):

        #     gcp_df = load_gcp_assumptions(segment=segment)
        #     pol_count_finance_df = load_pol_count_assumptions(segment=segment)
        #     gcp_per_pol = pd.merge(gcp_df
        #                             ,pol_count_finance_df
        #                             ,on=[DATE_SOLD_END_OF_MONTH,SEGMENT],how="left")
                
        #     gcp_per_pol[GCP_PER_POL] = gcp_per_pol[GCP] / gcp_per_pol[POLICY_COUNT]

        #     gcp_per_pol['finance_cutoff_date'] = finance_cutoff_date
        #     return gcp_per_pol[['finance_cutoff_date',SEGMENT,DATE_SOLD_END_OF_MONTH,GCP,POLICY_COUNT,GCP_PER_POL]]

        # we assume we have access to the gcp per pol from finance as well as the gcp and policy count
        # gcp_per_pol = get_gcp_per_pol_from_finance(segment,finance_cutoff_date)
        # condition_2 = (gcp_per_pol[DATE_SOLD_END_OF_MONTH]>=CUTOFF_DATE)
        # shocked_future_pol = gcp_per_pol[condition_2]

        # cohorts_app_to_dep_pol,fig_training_cohorts = get_training_cohorts(training_df=policies_seg)

        # month_diff_column_name = "app_to_depart_month"
        # level = 0.85
        # metric_column_name=POLICY_COUNT
        # metric_column_name_output = "idpol_unique_"
        # start_date_column = "dateApp"
        # end_date_column = "dateDepart"
        # past_futurename = "pol_past_present"
        # method_app_to_dep = "month"
    
        # r_future_pol_cohorts = main_pol_cohorts_development(df=policies_seg
        #                                                         ,end_date=CUTOFF_DATE
        #                                                         ,future_pol = shocked_future_pol
        #                                                         ,cohorts_past=cohorts_app_to_dep_pol
        #                                                         ,start_date_column=start_date_column
        #                                                         ,end_date_column=end_date_column
        #                                                         ,month_diff_column_name=month_diff_column_name
        #                                                         ,metric_column_name=metric_column_name
        #                                                         ,past_futurename = past_futurename
        #                                                         ,level=level
        #                                                         ,method = method_app_to_dep
        #                                                         ,show = True
        #                                                         ,metric_column_name_output=metric_column_name_output
        #                                                         )
        # past_future_pol_cohorts_df_ = r_future_pol_cohorts["data"]
        # new_past_future_pol_cohorts_df_ = past_future_pol_cohorts_df_.groupby(DATE_DEPART_END_OF_MONTH,as_index=False).agg({"idpol_unique_":"sum"})