import numpy as np
from functools import wraps
from sklearn.model_selection import KFold
import pandas as pd


def calculate_profit_decorator(strategy_func):
    @wraps(strategy_func)
    def wrapper(*args, **kwargs):
        # Initialize total profit

        # Execute the strategy function to get profit
        profit = strategy_func(*args, **kwargs)
        profit = np.array(profit)
        # Remove NaN values
        profit = profit[~np.isnan(profit)]
        total_profit = np.sum(profit)

        # Calculate total trading days
        n_trading_days = len(profit)

        # Print results
        print(
            f"Total Profit per year: {total_profit / n_trading_days * 365}, n_trading_days: {n_trading_days}"
        )
        print(
            "Increase percent per day: ",
            total_profit / n_trading_days / kwargs.get("input_dollar", 1),
        )

        return profit

    return wrapper


def cv_5fold(data_df, n_bins=20):
    """
    Generate train and test indices for 5-fold cross-validation.

    Parameters:
    - data_df: DataFrame containing the data to be split.
    - n_bins: Number of bins to split the data into.

    Yields:
    - train_index: Indices for the training data.
    - test_index: Indices for the test data.
    """
    n_data_points = len(data_df)
    n_data_bins = n_data_points // n_bins
    indices = np.arange(n_data_points)
    data_df_list = [
        indices[i : i + n_data_bins] for i in range(0, n_data_points, n_data_bins)
    ]
    data_df_list = np.array(data_df_list, dtype=object)
    cv = KFold(n_splits=5, shuffle=True)

    for train_bins, test_bins in cv.split(data_df_list):
        train_index = np.concatenate([data_df_list[i] for i in train_bins])
        test_index = np.concatenate([data_df_list[i] for i in test_bins])
        yield train_index, test_index


def cv_causal(data_df, train=180, test=90, step=90):
    """
    Generate train and test indices, using previous 'train' days as training data, and the next 'test' days as test data, step by 'step' days.

    Parameters:
    - data_df: DataFrame containing the data to be split.

    Yields:
    - train_index: Indices for the training data.
    - test_index: Indices for the test data.
    """
    # 统计有数据的时间段的个数，根据Date
    dates = data_df["Date"].unique()
    dates = np.sort(dates)
    n_dates = len(dates)
    for i in range(train, n_dates, step):
        train_dates = dates[i : i + train]
        test_dates = dates[i + train : i + train + test]
        train_index = data_df["Date"].isin(train_dates)
        test_index = data_df["Date"].isin(test_dates)
        yield train_index, test_index


def calc_spread_return_sharpe(
    df: pd.DataFrame, portfolio_size: int = 200, toprank_weight_ratio: float = 2
) -> float:
    """
    Calculate the Sharpe ratio based on spread return.

    Args:
        df (pd.DataFrame): Predicted results.
        portfolio_size (int): Number of equities to buy/sell.
        toprank_weight_ratio (float): The relative weight of the most highly ranked stock compared to the least.

    Returns:
        float: Sharpe ratio.
    """

    def _calc_spread_return_per_day(df, portfolio_size, toprank_weight_ratio):
        assert df["Rank"].min() == 0
        assert df["Rank"].max() == len(df["Rank"]) - 1
        weights = np.linspace(start=toprank_weight_ratio, stop=1, num=portfolio_size)
        purchase = (
            df.sort_values(by="Rank")["Target"][:portfolio_size] * weights
        ).sum() / weights.mean()
        short = (
            df.sort_values(by="Rank", ascending=False)["Target"][:portfolio_size]
            * weights
        ).sum() / weights.mean()
        return purchase - short

    buf = df.groupby("Date").apply(
        _calc_spread_return_per_day, portfolio_size, toprank_weight_ratio
    )
    sharpe_ratio = buf.mean() / buf.std()
    return sharpe_ratio
