"""
Configuration utilities for Claim Forecast application
Loads settings from environment variables with sensible defaults
"""

import os
from pathlib import Path
from typing import Optional

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed, will use system environment variables only
    pass


class Config:
    """Configuration management for Claim Forecast"""
    
    # Data paths
    #DATA_ROOT: str = os.getenv('CLAIM_FORECAST_DATA_ROOT', './data')
    BACKUP_ROOT: str = os.getenv('CLAIM_FORECAST_BACKUP_ROOT', './data/_data')
    
    # Backup mode paths
    BACKUP_MODE_CSA_PATH: str = os.getenv('BACKUP_MODE_CSA_PATH', f'{BACKUP_ROOT}/csa')
    BACKUP_MODE_TM_PATH: str = os.getenv('BACKUP_MODE_TM_PATH', f'{BACKUP_ROOT}/tripmate')
    
    # Specific forecast paths (with defaults)
    POLICY_FORECAST_PATH: str = os.getenv('POLICY_FORECAST_PATH', f'{BACKUP_ROOT}/policy_count_forecast')
    FREQUENCY_FORECAST_PATH: str = os.getenv('FREQUENCY_FORECAST_PATH', f'{BACKUP_ROOT}/frequency_forecast')
    CLAIM_FORECAST_PATH: str = os.getenv('CLAIM_FORECAST_PATH', f'{BACKUP_ROOT}/claim_count_forecast')
    
    # Application settings
    DEFAULT_BACKUP_MODE: bool = os.getenv('DEFAULT_BACKUP_MODE', 'false').lower() == 'true'
    DEFAULT_GRANULARITY: str = os.getenv('DEFAULT_GRANULARITY', 'Month')
    MAX_BATCH_SEGMENTS: int = int(os.getenv('MAX_BATCH_SEGMENTS', '50'))
    
    # Logging
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE_PATH: str = os.getenv('LOG_FILE_PATH', './logs/claim_forecast.log')
    
    # Advanced settings
    FORECAST_HORIZON_MONTHS: int = int(os.getenv('FORECAST_HORIZON_MONTHS', '36'))
    DEFAULT_CONFIDENCE_LEVEL: float = float(os.getenv('DEFAULT_CONFIDENCE_LEVEL', '0.95'))
    CACHE_TTL: int = int(os.getenv('CACHE_TTL', '3600'))
    ENABLE_PROFILING: bool = os.getenv('ENABLE_PROFILING', 'false').lower() == 'true'
    
    @classmethod
    def get_policy_results_path(cls) -> Path:
        """Get path to policy forecast results"""
        return Path(cls.POLICY_FORECAST_PATH) / "_results"
    
    @classmethod
    def get_frequency_results_path(cls) -> Path:
        """Get path to frequency forecast results"""
        return Path(cls.FREQUENCY_FORECAST_PATH)
    
    @classmethod
    def get_claim_results_path(cls) -> Path:
        """Get path to claim forecast results"""
        return Path(cls.CLAIM_FORECAST_PATH) / "_results"
    
    @classmethod
    def get_finance_input_path(cls) -> Path:
        """Get path to finance input data"""
        return Path(cls.POLICY_FORECAST_PATH) / "input_finance"
    
    @classmethod
    def get_config_path(cls, config_type: str) -> Path:
        """
        Get path to configuration file
        
        Args:
            config_type: Type of config ('lag', 'freq', etc.)
        
        Returns:
            Path to config file
        """
        if config_type == 'lag':
            return Path(cls.POLICY_FORECAST_PATH) / "config_lag.json"
        elif config_type == 'freq':
            return Path(cls.FREQUENCY_FORECAST_PATH) / "config_freq.json"
        else:
            raise ValueError(f"Unknown config type: {config_type}")
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist"""
        directories = [
            cls.DATA_ROOT,
            cls.BACKUP_ROOT,
            cls.BACKUP_MODE_CSA_PATH,
            cls.BACKUP_MODE_TM_PATH,
            cls.POLICY_FORECAST_PATH,
            cls.FREQUENCY_FORECAST_PATH,
            cls.CLAIM_FORECAST_PATH,
            cls.get_policy_results_path(),
            cls.get_frequency_results_path(),
            cls.get_claim_results_path(),
            cls.get_finance_input_path(),
            Path(cls.LOG_FILE_PATH).parent
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def validate_config(cls) -> tuple[bool, list[str]]:
        """
        Validate configuration
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check if critical paths exist
        if not Path(cls.DATA_ROOT).exists():
            issues.append(f"Data root does not exist: {cls.DATA_ROOT}")
        
        # Check if backup paths exist when backup mode is enabled
        if cls.DEFAULT_BACKUP_MODE:
            if not Path(cls.BACKUP_MODE_CSA_PATH).exists():
                issues.append(f"CSA backup path does not exist: {cls.BACKUP_MODE_CSA_PATH}")
            if not Path(cls.BACKUP_MODE_TM_PATH).exists():
                issues.append(f"TripMate backup path does not exist: {cls.BACKUP_MODE_TM_PATH}")
        
        # Check numeric ranges
        if cls.FORECAST_HORIZON_MONTHS < 1 or cls.FORECAST_HORIZON_MONTHS > 120:
            issues.append(f"Invalid forecast horizon: {cls.FORECAST_HORIZON_MONTHS} (must be 1-120)")
        
        if cls.DEFAULT_CONFIDENCE_LEVEL < 0.5 or cls.DEFAULT_CONFIDENCE_LEVEL > 0.999:
            issues.append(f"Invalid confidence level: {cls.DEFAULT_CONFIDENCE_LEVEL} (must be 0.5-0.999)")
        
        return len(issues) == 0, issues
    
    @classmethod
    def print_config(cls):
        """Print current configuration (for debugging)"""
        print("=" * 60)
        print("Claim Forecast Configuration")
        print("=" * 60)
        print(f"Data Root:              {cls.DATA_ROOT}")
        print(f"Backup Root:            {cls.BACKUP_ROOT}")
        print(f"Backup Mode:            {cls.DEFAULT_BACKUP_MODE}")
        print(f"CSA Backup Path:        {cls.BACKUP_MODE_CSA_PATH}")
        print(f"TripMate Backup Path:   {cls.BACKUP_MODE_TM_PATH}")
        print(f"Policy Forecast Path:   {cls.POLICY_FORECAST_PATH}")
        print(f"Frequency Path:         {cls.FREQUENCY_FORECAST_PATH}")
        print(f"Claim Forecast Path:    {cls.CLAIM_FORECAST_PATH}")
        print(f"Log Level:              {cls.LOG_LEVEL}")
        print(f"Forecast Horizon:       {cls.FORECAST_HORIZON_MONTHS} months")
        print("=" * 60)


# Singleton instance
config = Config()


# Convenience functions
def get_data_root() -> str:
    """Get data root path"""
    return config.DATA_ROOT


def get_backup_root() -> str:
    """Get backup root path"""
    return config.BACKUP_ROOT


def is_backup_mode() -> bool:
    """Check if backup mode is enabled"""
    return config.DEFAULT_BACKUP_MODE


if __name__ == "__main__":
    # Validate and print config when run directly
    config.print_config()
    
    is_valid, issues = config.validate_config()
    if is_valid:
        print("✅ Configuration is valid")
    else:
        print("❌ Configuration has issues:")
        for issue in issues:
            print(f"  - {issue}")

