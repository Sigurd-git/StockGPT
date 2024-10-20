# # Existing strategy functions
# @calculate_profit_decorator
# def mlp_sell(
#     data_df,
#     feature_names,
#     output_variable="Adj Close",
#     input_dollar=1,
#     seed=42,
#     stop_loss_pct=0.02,
#     take_profit_pct=0.05,
# ):
#     # Remove NaN values
#     data_df = data_df.dropna()

#     np.random.seed(seed)

#     cv = cv_causal(data_df)

#     profit = []
#     for train_index, test_index in cv:
#         train_data = data_df.iloc[train_index]
#         test_data = data_df.iloc[test_index]

#         # Get predictions
#         predictions = mlpnet(
#             train_data, test_data, feature_names, output_variable, device="mps"
#         )

#         # Calculate profit with enhanced strategy
#         test_output = test_data[output_variable]
#         test_open = test_data["Open"]

#         for i in range(len(predictions)):
#             open_price = test_open.iloc[i]
#             close_price = test_output.iloc[i]
#             sell_signal = predictions[i] < open_price

#             if sell_signal:
#                 potential_profit = input_dollar * (1 - close_price / open_price)

#                 if close_price > open_price * (1 + stop_loss_pct):
#                     potential_profit = -input_dollar * stop_loss_pct
#                 elif close_price < open_price * (1 - take_profit_pct):
#                     potential_profit = input_dollar * take_profit_pct

#                 profit.append(potential_profit)
#             else:
#                 profit.append(0)

#     return np.array(profit)


# 对每只股票进行特征提取和预测
all_predictions = []
for ticker in tickers:
    try:
        data_df = data[data["Ticker"] == ticker]
        data_df = extract_features(data_df)
        predictions = mlp_sell(
            data_df,
            [
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
            ],
            "Adj Close",
        )
        data_df["Prediction"] = predictions
        data_df["Target"] = data_df["Adj Close"].shift(-1)  # 使用下一天的收盘价作为目标
        data_df = data_df.dropna(subset=["Prediction", "Target"])
        data_df["Ticker"] = ticker
        all_predictions.append(data_df)
    except Exception as e:
        print(f"Error processing {ticker}: {e}")

# 合并所有股票的预测结果
all_predictions_df = pd.concat(all_predictions)

# 对股票进行排名
all_predictions_df["Rank"] = (
    all_predictions_df.groupby("Date")["Prediction"].rank(
        method="first", ascending=False
    )
    - 1
)
all_predictions_df["Rank"] = all_predictions_df["Rank"].astype(int)

# 计算 Sharpe 比率
sharpe_ratio = calc_spread_return_sharpe(all_predictions_df)
print(f"Sharpe Ratio for S&P 500: {sharpe_ratio}")
