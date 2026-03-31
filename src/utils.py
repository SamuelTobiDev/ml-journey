import pandas as pd
from typing import Optional

def load_data(url: str,num_rows: Optional[int] = None) -> pd.DataFrame:
    """Load data from a CSV file at the given URL. Optionally limit to a specified number of rows."""
    df = pd.read_csv(url, nrows=num_rows)
    return df

def compute_stats(df: pd.DataFrame)-> dict[str, float]:
    """Returns key stats about the dataset"""
    return {
        'survival_rate': df['Survived'].mean(),
        'avg_age': df['Age'].mean(),
        'avg_fare': df['Fare'].mean(),
    }


def split_data(df: pd.DataFrame, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits the dataset into training and testing sets."""
    split_index = int(len(df) * (1 - test_size))
    return df.iloc[:split_index], df.iloc[split_index:]