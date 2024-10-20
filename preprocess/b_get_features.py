import pandas as pd
import numpy as np


def accumulation_distribution(data_df):
    def calc_adl(group):
        clv = (
            (group["Adj Close"].shift(1) - group["Low"])
            - (group["High"] - group["Adj Close"].shift(1))
        ) / (group["High"] - group["Low"])
        group["ADL"] = (clv * group["Volume"].shift(1)).cumsum()
        return group

    return data_df.groupby("Ticker", group_keys=False).apply(calc_adl)


def atr(data_df, window=14):
    def calc_atr(group):
        high_low = group["High"] - group["Low"]
        high_close = abs(group["High"] - group["Adj Close"].shift(1))
        low_close = abs(group["Low"] - group["Adj Close"].shift(1))
        true_range = high_low.combine(high_close, max).combine(low_close, max)
        group["ATR"] = (
            true_range.rolling(window, min_periods=1).mean().shift(1)
        )  # Ensure past data is used
        return group

    return data_df.groupby("Ticker", group_keys=False).apply(calc_atr)


def bollinger_bands(data_df, window=20):
    def calc_bollinger(group):
        rolling_mean = group["Adj Close"].shift(1).rolling(window, min_periods=1).mean()
        rolling_std = group["Adj Close"].shift(1).rolling(window, min_periods=1).std()
        group["Bollinger_Upper"] = rolling_mean + (rolling_std * 2)
        group["Bollinger_Lower"] = rolling_mean - (rolling_std * 2)
        return group

    return data_df.groupby("Ticker", group_keys=False).apply(calc_bollinger)


def cci(data_df, window=20):
    def calc_cci(group):
        typical_price = (
            group["High"].shift(1) + group["Low"].shift(1) + group["Adj Close"].shift(1)
        ) / 3
        moving_average = typical_price.rolling(window, min_periods=1).mean()
        mean_deviation = typical_price.rolling(window, min_periods=1).apply(
            lambda x: np.mean(np.abs(x - np.mean(x)))
        )
        group["CCI"] = (typical_price - moving_average) / (0.015 * mean_deviation)
        return group

    return data_df.groupby("Ticker", group_keys=False).apply(calc_cci)


def macd(data_df, short_window=12, long_window=26, signal_window=9):
    def calc_macd(group):
        group["EMA12"] = (
            group["Adj Close"].shift(1).ewm(span=short_window, adjust=False).mean()
        )
        group["EMA26"] = (
            group["Adj Close"].shift(1).ewm(span=long_window, adjust=False).mean()
        )
        group["MACD"] = group["EMA12"] - group["EMA26"]
        group["MACD_Signal"] = (
            group["MACD"].ewm(span=signal_window, adjust=False).mean()
        )
        group["MACD_Hist"] = group["MACD"] - group["MACD_Signal"]
        return group

    return data_df.groupby("Ticker", group_keys=False).apply(calc_macd)


def ma_1(data_df):
    def calc_ma_1(group):
        group["ma_1"] = (
            group["Adj Close"].shift(1).rolling(window=1, min_periods=1).mean()
        )
        return group

    return data_df.groupby("Ticker", group_keys=False).apply(calc_ma_1)


def ma_120(data_df):
    def calc_ma_120(group):
        group["ma_120"] = (
            group["Adj Close"].shift(1).rolling(window=120, min_periods=1).mean()
        )
        return group

    return data_df.groupby("Ticker", group_keys=False).apply(calc_ma_120)


def ma_20(data_df):
    def calc_ma_20(group):
        group["ma_20"] = (
            group["Adj Close"].shift(1).rolling(window=20, min_periods=1).mean()
        )
        return group

    return data_df.groupby("Ticker", group_keys=False).apply(calc_ma_20)


def ma_30(data_df):
    def calc_ma_30(group):
        group["ma_30"] = (
            group["Adj Close"].shift(1).rolling(window=30, min_periods=1).mean()
        )
        return group

    return data_df.groupby("Ticker", group_keys=False).apply(calc_ma_30)


def ma_50(data_df):
    def calc_ma_50(group):
        group["ma_50"] = (
            group["Adj Close"].shift(1).rolling(window=50, min_periods=1).mean()
        )
        return group

    return data_df.groupby("Ticker", group_keys=False).apply(calc_ma_50)


def ma_60(data_df):
    def calc_ma_60(group):
        group["ma_60"] = (
            group["Adj Close"].shift(1).rolling(window=60, min_periods=1).mean()
        )
        return group

    return data_df.groupby("Ticker", group_keys=False).apply(calc_ma_60)


def momentum(data_df, window=10):
    def calc_momentum(group):
        group["Momentum"] = group["Adj Close"].shift(1).diff(window)
        return group

    return data_df.groupby("Ticker", group_keys=False).apply(calc_momentum)


def price_rate_of_change(data_df, window=12):
    def calc_roc(group):
        group["ROC"] = (
            group["Adj Close"].shift(1).diff(window)
            / group["Adj Close"].shift(window + 1)
        ) * 100
        return group

    return data_df.groupby("Ticker", group_keys=False).apply(calc_roc)


def relative_volume(data_df, window=20):
    def calc_rvol(group):
        group["RVOL"] = (
            group["Volume"].shift(1)
            / group["Volume"].shift(1).rolling(window, min_periods=1).mean()
        )
        return group

    return data_df.groupby("Ticker", group_keys=False).apply(calc_rvol)


def rsi(data_df, window=14):
    def calc_rsi(group):
        delta = group["Adj Close"].diff(1).shift(1)  # Prevent future data leakage
        gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
        rs = gain / loss
        group["RSI"] = 100 - (100 / (1 + rs))
        return group

    return data_df.groupby("Ticker", group_keys=False).apply(calc_rsi)


def stochastic_oscillator(data_df, window=14):
    def calc_stochastic(group):
        low_min = group["Low"].shift(1).rolling(window, min_periods=1).min()
        high_max = group["High"].shift(1).rolling(window, min_periods=1).max()
        group["Stochastic_K"] = (
            (group["Adj Close"].shift(1) - low_min) / (high_max - low_min) * 100
        )
        group["Stochastic_D"] = (
            group["Stochastic_K"].rolling(3, min_periods=1).mean().shift(1)
        )
        return group

    return data_df.groupby("Ticker", group_keys=False).apply(calc_stochastic)


def stoch_rsi(data_df, window=14):
    def calc_stoch_rsi(group):
        # Calculate RSI
        delta = group["Adj Close"].diff(1).shift(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        # Calculate Stochastic RSI
        min_rsi = rsi.rolling(window, min_periods=1).min().shift(1)
        max_rsi = rsi.rolling(window, min_periods=1).max().shift(1)
        group["Stoch_RSI"] = (rsi - min_rsi) / (max_rsi - min_rsi)
        return group

    return data_df.groupby("Ticker", group_keys=False).apply(calc_stoch_rsi)


def vroc(data_df, window=14):
    def calc_vroc(group):
        group["VROC"] = (
            group["Volume"].shift(1).diff(window) / group["Volume"].shift(window + 1)
        ) * 100
        return group

    return data_df.groupby("Ticker", group_keys=False).apply(calc_vroc)


def williams_r(data_df, window=14):
    def calc_williams_r(group):
        high_max = group["High"].shift(1).rolling(window, min_periods=1).max()
        low_min = group["Low"].shift(1).rolling(window, min_periods=1).min()
        group["Williams_R"] = (
            (high_max - group["Adj Close"].shift(1)) / (high_max - low_min) * -100
        )
        return group

    return data_df.groupby("Ticker", group_keys=False).apply(calc_williams_r)


def extract_features(data_df):
    data_df = accumulation_distribution(data_df)
    data_df = atr(data_df)
    data_df = bollinger_bands(data_df)
    data_df = cci(data_df)
    data_df = macd(data_df)
    data_df = ma_1(data_df)
    data_df = ma_20(data_df)
    data_df = ma_30(data_df)
    data_df = ma_50(data_df)
    data_df = ma_60(data_df)
    data_df = ma_120(data_df)
    data_df = momentum(data_df)
    data_df = price_rate_of_change(data_df)
    data_df = relative_volume(data_df)
    data_df = rsi(data_df)
    data_df = stochastic_oscillator(data_df)
    data_df = stoch_rsi(data_df)
    data_df = vroc(data_df)
    data_df = williams_r(data_df)
    return data_df


if __name__ == "__main__":
    data_df = pd.read_parquet("data/sp500_data.parquet")
    data_df = extract_features(data_df)
    data_df.to_parquet("data/sp500_data_with_features.parquet")
