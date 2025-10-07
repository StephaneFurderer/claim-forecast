
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

from frequency_development.constants import *
def plot_app_to_dep_policy_cohorts(cohorts_df:pd.DataFrame,metric:str=POLICY_COUNT)-> pd.DataFrame:
    # Creating an area chart
    for col in [DATE_SOLD_END_OF_MONTH,DATE_DEPART_END_OF_MONTH]:
        cohorts_df[f"{col}_str"] = cohorts_df[col].astype("str")

    fig = px.area(cohorts_df, x=f"{DATE_DEPART_END_OF_MONTH}_str", y=metric, color=f"{DATE_SOLD_END_OF_MONTH}_str",
                labels={
                    f"{DATE_DEPART_END_OF_MONTH}_str": "Departure Date",
                    metric: "Volume of Policies",
                    f"{DATE_SOLD_END_OF_MONTH}_str": "Application Date"
                },
                title="Volume of Policies by Application and Departure Date")
    return fig

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