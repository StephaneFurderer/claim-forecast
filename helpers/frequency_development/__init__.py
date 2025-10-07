"""
Frequency Development Analysis Package
"""

from .core import (
    analyze_and_save_segment,
    calculate_development_metrics,
    compute_claim_development_triangle_pandas,
    compute_development_factors,
    compute_frequency_dev_pandas,
    compute_raw_average_ultimate_frequency,
    compute_reported_frequency,
    estimate_ultimate_frequencies,
    filter_and_group_by_period,
    flag_major_cats_pandas,
    get_manual_overrides,
    load_data,
    load_user_config,
    project_cohort_development,
    save_user_config,
    select_best_frequency,
    set_manual_overrides,
    preprocess_data,
    get_or_create_config_for_key,
    load_data_backup,
    load_data_backup_tripmate
)

from .plot_utils import (
    plot_development_factors,
    plot_forecast_trend,
    plot_best_frequency
)

__version__ = "0.1.6" 