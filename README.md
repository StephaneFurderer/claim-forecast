# 📊 Claim Count Forecast - Long Term Model

A comprehensive insurance claims forecasting system that combines policy volume projections, frequency development patterns, and timing distributions to produce multi-year claim forecasts.

## 🎯 Overview

Three-stage forecasting pipeline:

1. **Policy Count Forecast** - Project future policy volumes by departure date
2. **Frequency Development** - Estimate ultimate claims per policy using chain-ladder techniques
3. **Claim Count Forecast** - Combine policy volumes × frequencies to forecast claim counts

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Historical policy and claims data

### Installation

```bash
# 1. Clone repository
<link to bitbucket here>
cd <repo>

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment (copy .env.example to .env and edit paths)
cp .env.example .env
nano .env

# 4. Launch
streamlit run app/welcome.py
```

### Data Directory Structure

The app expects this structure (created automatically or manually):

```
data/
├── _data/                    # Backup CSV data
│   ├── csa/YYYY-MM-DD/
│   └── tripmate/YYYY-MM-DD/
├── policy_count_forecast/
│   ├── _results/
│   └── input_finance/
├── frequency_forecast/
└── claim_count_forecast/
    └── _results/
```

## 📖 User Guide

### Workflow

**Stage 1: Policy Count Forecast**
- Select segment and cutoff dates
- Review forecast charts
- Save results

**Stage 2: Frequency Development**  
- Select segment
- Exclude catastrophic events if needed
- Apply manual overrides (optional)
- Save best frequencies

**Stage 3: Claim Count Forecast**
- Select scenario and segment
- Review combined forecast
- Save final results

Each stage builds on the previous. The welcome page provides detailed guidance.

### Configuration

**Scenarios** (`app/scenarios.json`): Define comparison periods
**Segments**: CSA (Airbnb, Booking.com, etc.) and TripMate
**Environment** (`.env`): Set data paths and preferences

## 🔧 Configuration

### Key Environment Variables (`.env`)

| Variable | Description | Default |
|----------|-------------|---------|
| `CLAIM_FORECAST_DATA_ROOT` | Root data directory | `./data` |
| `DEFAULT_BACKUP_MODE` | Use CSV files instead of DB | `true` |
| `LOG_LEVEL` | Logging verbosity | `INFO` |

### Config Files

- `config_lag.json` - Lag settings per segment/date
- `config_freq.json` - Frequency exclusions and overrides  
- `scenarios.json` - Scenario definitions

## 📁 Project Structure

```
claim-forecast/
├── app/
│   ├── welcome.py                          # Main entry point
│   ├── pages/
│   │   ├── 1_dashboard_policy_count_forecast.py
│   │   ├── 2_dashboard_frequency_development.py
│   │   └── 3_dashboard_claim_count.py
│   └── scenarios.json                      # Scenario definitions
├── helpers/
│   ├── policy_count_forecast/              # Policy forecasting logic
│   │   ├── core.py
│   │   └── plot_utils.py
│   ├── frequency_development/              # Frequency development logic
│   │   ├── core.py
│   │   ├── constants.py
│   │   └── plot_utils.py
│   └── claim_count_forecast/               # Claim forecasting logic
│       ├── core.py
│       └── aoc_core.py
├── .env.example                            # Environment template
├── .env                                    # Local config (gitignored)
├── requirements.txt                        # Python dependencies
└── README.md                               # This file
```

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| Data not loading | Check `.env` paths, enable backup mode |
| Missing segments | Verify segment names match in policies/claims |
| Slow performance | Use backup mode, process individually |
| Wrong forecast | Validate input data, check finance assumptions |

## 💡 Best Practices

- Complete all 3 stages per segment before moving to next
- Keep backup CSVs updated
- Document exclusions and overrides
- Validate against actuals regularly
- Run forecasts consistently (monthly)

## 📧 Support

Contact the actuarial team or review logs in `./logs/claim_forecast.log`

---

*Built for actuaries, by actuaries*

