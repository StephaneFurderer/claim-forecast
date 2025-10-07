import streamlit as st
import os
import pandas as pd
from pathlib import Path
import json

st.set_page_config(
    page_title="Claim Forecast - Welcome",
    page_icon="📊",
    layout="wide"
)

# Title and introduction
st.title("📊 Claim Count Forecast: Long Term Model")
st.markdown("### Welcome to the Claims Forecasting Platform")

st.markdown("""
This application provides a comprehensive framework for forecasting insurance claim counts 
over multi-year horizons. The system combines policy volume projections, frequency development 
patterns, and timing distributions to produce accurate claim forecasts.
""")

# Workflow section
st.markdown("---")
st.header("🔄 Forecasting Workflow")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 1️⃣ Policy Count Forecast
    **What it does:**
    - Projects future policy volumes by departure date
    - Incorporates financial assumptions (GCP, policy counts)
    - Applies seasonal patterns and YoY growth rates
    
    **When to use:**
    - Start here for any new forecast period
    - Update when financial assumptions change
    
    **Key inputs:**
    - Historical policy data
    - Finance projections
    - Growth rate assumptions
    """)
    
    if st.button("🏠 Go to Policy Forecast", use_container_width=True):
        st.switch_page("pages/1_dashboard_policy_count_forecast.py")

with col2:
    st.markdown("""
    ### 2️⃣ Frequency Development
    **What it does:**
    - Estimates ultimate claims per policy
    - Uses chain-ladder development technique
    - Accounts for catastrophic events
    
    **When to use:**
    - After policy forecast is complete
    - Update when claim patterns shift
    
    **Key inputs:**
    - Historical claims data
    - Policy volumes
    - Development assumptions
    """)
    
    if st.button("📈 Go to Frequency Forecast", use_container_width=True):
        st.switch_page("pages/2_dashboard_frequency_development.py")

with col3:
    st.markdown("""
    ### 3️⃣ Claim Count Forecast
    **What it does:**
    - Combines policy volumes × frequency
    - Projects claims by departure & received date
    - Handles timing patterns and lags
    
    **When to use:**
    - After both policy and frequency are ready
    - Final step before reporting
    
    **Key outputs:**
    - Claims by departure date
    - Claims by received date (for cash flow)
    - Full cohort matrices
    """)
    
    if st.button("🎯 Go to Claim Forecast", use_container_width=True):
        st.switch_page("pages/3_dashboard_claim_count.py")

# Status dashboard
st.markdown("---")
st.header("📋 Forecast Status")

# Check for existing results
try:
    # Get data root from environment variable
    data_root = os.getenv('CLAIM_FORECAST_DATA_ROOT', './data')
    
    # Check if results exist
    policy_results_path = Path(data_root) / "policy_count_forecast" / "_results"
    freq_results_path = Path(data_root) / "frequency_forecast"
    claim_results_path = Path(data_root) / "claim_count_forecast" / "_results"
    
    status_data = []
    
    # Policy forecast status
    if policy_results_path.exists():
        pol_file = policy_results_path / "pol_count_per_dep_.csv"
        if pol_file.exists():
            pol_df = pd.read_csv(pol_file)
            pol_df['cutoff'] = pd.to_datetime(pol_df['cutoff'])
            latest_pol = pol_df['cutoff'].max()
            segments_pol = pol_df['segment'].nunique()
            status_data.append({
                "Stage": "Policy Forecast",
                "Status": "✅ Complete",
                "Latest Date": latest_pol.strftime('%Y-%m-%d') if pd.notna(latest_pol) else "N/A",
                "Segments": segments_pol
            })
        else:
            status_data.append({
                "Stage": "Policy Forecast",
                "Status": "⏳ Pending",
                "Latest Date": "N/A",
                "Segments": 0
            })
    else:
        status_data.append({
            "Stage": "Policy Forecast",
            "Status": "❌ Not Started",
            "Latest Date": "N/A",
            "Segments": 0
        })
    
    # Frequency forecast status
    if freq_results_path.exists():
        freq_file = freq_results_path / "best_frequencies.csv"
        if freq_file.exists():
            freq_df = pd.read_csv(freq_file)
            freq_df['cutoff'] = pd.to_datetime(freq_df['cutoff'])
            latest_freq = freq_df['cutoff'].max()
            segments_freq = freq_df['segment'].nunique()
            status_data.append({
                "Stage": "Frequency Development",
                "Status": "✅ Complete",
                "Latest Date": latest_freq.strftime('%Y-%m-%d') if pd.notna(latest_freq) else "N/A",
                "Segments": segments_freq
            })
        else:
            status_data.append({
                "Stage": "Frequency Development",
                "Status": "⏳ Pending",
                "Latest Date": "N/A",
                "Segments": 0
            })
    else:
        status_data.append({
            "Stage": "Frequency Development",
            "Status": "❌ Not Started",
            "Latest Date": "N/A",
            "Segments": 0
        })
    
    # Claim forecast status
    if claim_results_path.exists():
        claim_file = claim_results_path / "final_claims_rec_data.csv"
        if claim_file.exists():
            claim_df = pd.read_csv(claim_file)
            claim_df['cutoff'] = pd.to_datetime(claim_df['cutoff'])
            latest_claim = claim_df['cutoff'].max()
            segments_claim = claim_df['segment'].nunique()
            status_data.append({
                "Stage": "Claim Count Forecast",
                "Status": "✅ Complete",
                "Latest Date": latest_claim.strftime('%Y-%m-%d') if pd.notna(latest_claim) else "N/A",
                "Segments": segments_claim
            })
        else:
            status_data.append({
                "Stage": "Claim Count Forecast",
                "Status": "⏳ Pending",
                "Latest Date": "N/A",
                "Segments": 0
            })
    else:
        status_data.append({
            "Stage": "Claim Count Forecast",
            "Status": "❌ Not Started",
            "Latest Date": "N/A",
            "Segments": 0
        })
    
    if status_data:
        status_df = pd.DataFrame(status_data)
        st.dataframe(status_df, use_container_width=True, hide_index=True)
    else:
        st.info("📌 No forecast results found. Start by running the Policy Count Forecast.")

except Exception as e:
    st.warning(f"⚠️ Could not load forecast status. Data directory may not be configured correctly.")
    st.info("💡 **Tip:** Set the `CLAIM_FORECAST_DATA_ROOT` environment variable to point to your data directory.")
    with st.expander("Technical Details"):
        st.code(str(e))

# Quick start guide
st.markdown("---")
st.header("🚀 Quick Start Guide")

with st.expander("📖 First Time Setup", expanded=False):
    st.markdown("""
    ### Initial Configuration
    
    1. **Configure Environment Variables**
       - Copy `.env.example` to `.env`
       - Update `CLAIM_FORECAST_DATA_ROOT` with your data directory path
       - Update `CLAIM_FORECAST_BACKUP_ROOT` if using backup mode
    
    2. **Verify Data Access**
       - Ensure you have access to policy and claims data
       - Check that backup CSVs exist if using backup mode
    
    3. **Run First Forecast**
       - Start with Policy Count Forecast for one segment
       - Proceed to Frequency Development
       - Complete with Claim Count Forecast
    
    4. **Save Results**
       - Use "Save Results" buttons to persist forecasts
       - Results are stored in CSV format for downstream use
    """)

with st.expander("💡 Tips & Best Practices", expanded=False):
    st.markdown("""
    ### Forecasting Best Practices
    
    - **Work Segment by Segment**: Complete all three stages for one segment before moving to the next
    - **Document Assumptions**: Use the exclude dates and manual override features to document decisions
    - **Version Control**: Each forecast is tagged with cutoff dates for traceability
    - **Validate Results**: Compare policy volumes and frequencies against actuals before proceeding
    - **Lag Parameter**: Use lag for segments with known reporting delays (typically TripMate)
    - **Backup Mode**: Enable backup mode when data warehouse is unavailable
    """)

with st.expander("🔧 Troubleshooting", expanded=False):
    st.markdown("""
    ### Common Issues
    
    **Data Not Loading:**
    - Check environment variables are set correctly
    - Verify data directory paths exist
    - Ensure CSV files are in expected format
    
    **Missing Segments:**
    - Check segment names match between policy and claims data
    - Verify date ranges have sufficient data
    
    **Slow Performance:**
    - Enable backup mode to use pre-processed CSVs
    - Process segments individually rather than "Save All"
    - Check data volume and consider filtering historical data
    
    **Results Not Saving:**
    - Verify write permissions on results directories
    - Check disk space availability
    - Ensure results path is correctly configured
    """)

# Footer
st.markdown("---")
st.markdown("""
### 📚 Additional Resources

- **Documentation**: See README.md in the project root
- **Support**: Contact the actuarial team for assistance
- **Updates**: Check the project repository for latest features

---
*Claim Forecast Platform v1.0 - Built with Streamlit*
""")
