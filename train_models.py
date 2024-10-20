import pandas as pd
from models.mlp import mlpnet
from models.utils import cv_causal


if __name__ == "__main__":
    data_df = pd.read_parquet("data/sp500_data_with_features.parquet")
    data_df = data_df.dropna()
    cv = cv_causal(data_df)
    feature_names = [
        "ADL",
        "ATR",
        "Bollinger_Upper",
        "Bollinger_Lower",
        "CCI",
        "MACD_Hist",
        "ma_1",
        "ma_20",
        "ma_30",
        "ma_50",
        "ma_60",
        "ma_120",
        "Momentum",
        "ROC",
        "RVOL",
        "RSI",
        "Stochastic_K",
        "Stochastic_D",
        "Stoch_RSI",
        "VROC",
        "Williams_R",
        "Open",
    ]
    output_variable = "Adj Close"
    mlpnet(data_df, cv, feature_names, output_variable)
