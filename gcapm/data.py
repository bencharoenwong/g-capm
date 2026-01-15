"""
Data loading and validation utilities.

This module provides clean, validated data loading for all datasets in the g-capm repository.
All data is pre-cleaned and standardized (ISO dates, snake_case columns).

Examples
--------
>>> import gcapm
>>> df = gcapm.load_data('returns_daily')
>>> df.head()
"""

import pandas as pd
import json
from pathlib import Path
from typing import Optional, List, Dict, Union

# Paths
MODULE_DIR = Path(__file__).parent
DATA_DIR = MODULE_DIR.parent / "data" / "cleaned"
METADATA_PATH = MODULE_DIR.parent / "data" / "metadata.json"


def load_metadata() -> Dict:
    """
    Load dataset metadata.

    Returns
    -------
    dict
        Metadata for all datasets including column descriptions, date ranges, etc.

    Examples
    --------
    >>> meta = gcapm.data.load_metadata()
    >>> meta['datasets']['returns_daily']['description']
    """
    with open(METADATA_PATH) as f:
        return json.load(f)


def list_datasets() -> List[str]:
    """
    List all available datasets.

    Returns
    -------
    list of str
        Names of available datasets

    Examples
    --------
    >>> import gcapm
    >>> gcapm.list_datasets()
    ['tsla_daily', 'tsla_monthly', 'mkt_timing', ...]
    """
    metadata = load_metadata()
    return sorted(metadata['datasets'].keys())


def load_data(dataset_name: str,
              parse_dates: bool = True,
              validate: bool = True) -> pd.DataFrame:
    """
    Load a cleaned dataset with optional validation.

    All datasets are pre-cleaned with:
    - ISO 8601 dates (YYYY-MM-DD)
    - snake_case column names
    - Sorted by date
    - No BOM or special characters

    Parameters
    ----------
    dataset_name : str
        Name of dataset to load. Use list_datasets() to see options.
        Valid names: 'tsla_daily', 'returns_daily', 'mkt_timing', etc.
    parse_dates : bool, default True
        Whether to parse 'date' column as datetime
    validate : bool, default True
        Whether to run data quality checks

    Returns
    -------
    pd.DataFrame
        The loaded dataset

    Raises
    ------
    FileNotFoundError
        If dataset name is invalid
    ValueError
        If validation fails (missing/infinite values, unsorted dates)

    Examples
    --------
    >>> import gcapm
    >>> df = gcapm.load_data('returns_daily')
    >>> df.head()
    >>>
    >>> # Load without parsing dates
    >>> df = gcapm.load_data('mkt_timing', parse_dates=False)
    >>>
    >>> # Load without validation (faster, for trusted data)
    >>> df = gcapm.load_data('tsla_daily', validate=False)
    """
    # Construct file path
    file_path = DATA_DIR / f"{dataset_name}.csv"

    if not file_path.exists():
        available = list_datasets()
        raise FileNotFoundError(
            f"Dataset '{dataset_name}' not found.\n"
            f"Available datasets: {available}"
        )

    # Load data
    df = pd.read_csv(file_path)

    # Parse dates
    if parse_dates and 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

    # Validate
    if validate:
        _validate_data(df, dataset_name)

    return df


def _validate_data(df: pd.DataFrame, dataset_name: str):
    """
    Validate data quality.

    Checks for:
    - Missing values (raises error)
    - Infinite values (raises error)
    - Date ordering if date column exists (raises error if not sorted)
    """
    # Check for missing values
    missing = df.isnull().sum()
    if missing.any():
        missing_cols = missing[missing > 0]
        raise ValueError(
            f"Missing values found in '{dataset_name}':\n{missing_cols}"
        )

    # Check for infinite values in numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        if (df[col] == float('inf')).any() or (df[col] == float('-inf')).any():
            raise ValueError(
                f"Infinite values found in '{dataset_name}' column '{col}'"
            )

    # Check date ordering if date column exists
    if 'date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['date']):
        if not df['date'].is_monotonic_increasing:
            raise ValueError(
                f"Dates in '{dataset_name}' are not in ascending order"
            )


class DataLoader:
    """
    Class-based interface for loading and managing multiple datasets.

    This is useful when you need to work with multiple datasets and want to
    keep them organized in one place.

    Attributes
    ----------
    datasets : dict
        Dictionary of loaded DataFrames, keyed by dataset name

    Examples
    --------
    >>> loader = gcapm.DataLoader()
    >>> loader.load('tsla_daily')
    >>> loader.load('returns_daily')
    >>>
    >>> # Access loaded data
    >>> loader.datasets['tsla_daily'].head()
    >>>
    >>> # Get info about all loaded datasets
    >>> loader.summary()
    """

    def __init__(self):
        self.datasets = {}
        self.metadata = load_metadata()

    def load(self, dataset_name: str, **kwargs) -> pd.DataFrame:
        """
        Load a dataset and store it.

        Parameters
        ----------
        dataset_name : str
            Name of dataset
        **kwargs
            Additional arguments passed to load_data()

        Returns
        -------
        pd.DataFrame
            The loaded dataset
        """
        df = load_data(dataset_name, **kwargs)
        self.datasets[dataset_name] = df
        return df

    def get(self, dataset_name: str) -> pd.DataFrame:
        """
        Get a previously loaded dataset.

        Parameters
        ----------
        dataset_name : str
            Name of dataset

        Returns
        -------
        pd.DataFrame
            The dataset

        Raises
        ------
        KeyError
            If dataset hasn't been loaded yet
        """
        if dataset_name not in self.datasets:
            raise KeyError(
                f"Dataset '{dataset_name}' not loaded yet. "
                f"Use loader.load('{dataset_name}') first."
            )
        return self.datasets[dataset_name]

    def summary(self) -> pd.DataFrame:
        """
        Get summary of all loaded datasets.

        Returns
        -------
        pd.DataFrame
            Summary with rows, columns, date range for each dataset
        """
        summaries = []

        for name, df in self.datasets.items():
            summary = {
                'dataset': name,
                'rows': len(df),
                'columns': len(df.columns),
            }

            if 'date' in df.columns:
                summary['date_min'] = df['date'].min()
                summary['date_max'] = df['date'].max()

            summaries.append(summary)

        return pd.DataFrame(summaries)

    def __repr__(self):
        n_loaded = len(self.datasets)
        return f"DataLoader({n_loaded} datasets loaded)"
