import yfinance as yf
import pandas as pd
import os
import pandas_market_calendars as mcal
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup


def get_metrics(
    start_time="2024-01-01",
    end_time="2024-05-13",
    save_dir="data",
    force_download=False,
):
    # Create cache directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    tickers = get_sp500_tickers()
    # Define cache file path
    cache_file = os.path.join(save_dir, f"sp500_data.parquet")

    all_days = mcal.get_calendar("NYSE").schedule(start_time, end_time).index

    if os.path.exists(cache_file) and not force_download:
        # Load cached data
        cached_data = pd.read_parquet(cache_file)

        # Identify non-trading days
        missing_days = all_days[~all_days.isin(cached_data.index.get_level_values(0))]

        if missing_days.to_numpy().any():
            # Download the missing data for each ticker
            new_datas = []
            for ticker in tickers:
                if len(missing_days) <= 10:
                    new_data = [
                        yf.download(
                            ticker,
                            start=day.strftime("%Y-%m-%d"),
                            end=(day + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                        )
                        for day in missing_days
                    ]
                else:
                    new_data = [
                        yf.download(
                            ticker,
                            start=missing_days[0].strftime("%Y-%m-%d"),
                            end=missing_days[-1].strftime("%Y-%m-%d"),
                        )
                    ]
                new_data = pd.concat(new_data)
                new_data["Ticker"] = ticker
                new_datas.append(new_data)

            # Combine the new data with the cached data
            combined_data = (
                pd.concat([cached_data] + new_datas).drop_duplicates().sort_index()
            )

            # Save the combined data to cache
            combined_data.to_parquet(cache_file)

            data = combined_data
        else:
            data = cached_data
            data = data.loc[(slice(start_time, end_time), slice(None)), :]
    else:
        # Download the data if no cache exists
        new_datas = []
        for ticker in tickers:
            data = yf.download(
                ticker,
                start=pd.Timestamp(start_time),
                end=(pd.Timestamp(end_time) + pd.Timedelta(days=1)),
                interval="1d",
            )
            data["Ticker"] = ticker
            new_datas.append(data)

        data = pd.concat(new_datas)

        # Save the data to cache
        data.to_parquet(cache_file)

    # Select Open, High, Low, Close, Volume
    data = data[["Open", "High", "Low", "Adj Close", "Volume", "Ticker"]]

    # fill NA to dates in the middle
    date_range = pd.date_range(start=start_time, end=end_time)
    data = data.set_index(["Ticker", data.index])
    data = (
        data.unstack(level=0)
        .reindex(date_range)
        .stack(level=0)
        .reset_index(level=0, drop=True)
    )

    return data


def get_metrics_7day(company_id, start_time, end_time, cache_dir="data", interval="1m"):
    """
    Fetch minute-level stock data for a given company within a 7-day range.

    Parameters:
    company_id (str): The stock ticker symbol.
    start_time (str): The start date in 'YYYY-MM-DD' format.
    end_time (str): The end date in 'YYYY-MM-DD' format.
    cache_dir (str): Directory to store cached data.
    interval (str): Data interval (default is '1m' for minute-level data).

    Returns:
    DataFrame: Stock data with Open, High, Low, Close, Adj Close, and Volume columns.
    """
    cache_file = f"{cache_dir}/{company_id}_{start_time}_{end_time}_{interval}.parquet"

    if os.path.exists(cache_file):
        cached_data = pd.read_parquet(cache_file)
        cached_data.index = pd.to_datetime(cached_data.index)
        cached_data = cached_data.tz_localize(None)

        missing_days = pd.date_range(start=start_time, end=end_time).difference(
            cached_data.index.normalize()
        )

        if not missing_days.empty:
            if len(missing_days) <= 10:
                new_datas = [
                    yf.download(
                        company_id,
                        start=day.strftime("%Y-%m-%d"),
                        end=(day + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                        interval=interval,
                    )
                    for day in missing_days
                ]
            else:
                new_datas = [
                    yf.download(
                        company_id,
                        start=missing_days[0].strftime("%Y-%m-%d"),
                        end=missing_days[-1].strftime("%Y-%m-%d"),
                        interval=interval,
                    )
                ]

            combined_data = (
                pd.concat([cached_data] + new_datas).drop_duplicates().sort_index()
            )

            combined_data.to_parquet(cache_file)

            data = combined_data
        else:
            data = cached_data
            data = data.loc[start_time:end_time]
    else:
        data = yf.download(
            company_id,
            start=pd.Timestamp(start_time),
            end=(pd.Timestamp(end_time) + pd.Timedelta(days=1)),
            interval=interval,
        )

        data.to_parquet(cache_file)

    data = data[["Open", "High", "Low", "Adj Close", "Volume"]]

    return data


# 添加获取 S&P 500 股票列表的函数
def get_sp500_tickers():
    """
    Fetch the list of S&P 500 tickers from Wikipedia.
    Returns:
        list: List of ticker symbols.
    """

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "lxml")
    table = soup.find("table", {"id": "constituents"})
    tickers = []
    for row in table.findAll("tr")[1:]:
        ticker = row.findAll("td")[0].text.strip()
        ticker = ticker.replace(".", "-")
        tickers.append(ticker)
    return tickers


def download_and_save_sp500_data(save_path="data/sp500_data.parquet"):
    """
    Download historical data for all S&P 500 companies and save it to a parquet file.

    This function fetches the list of S&P 500 tickers from Wikipedia, sets the date range for the past three years,
    downloads the historical stock data for these tickers using yfinance, and saves the data to a specified parquet file.

    Parameters:
    - save_path (str): The file path where the downloaded data will be saved in parquet format.

    Returns:
    - None
    """
    # Fetch S&P 500 tickers
    tickers = get_sp500_tickers()

    # Set date range
    today = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=3 * 365)).strftime("%Y-%m-%d")

    # Download data for all tickers
    data = yf.download(tickers, start=start_date, end=today)
    data = data.stack(level=1).rename_axis(["Date", "Ticker"]).reset_index()

    # Save data to parquet file
    data.to_parquet(save_path)


if __name__ == "__main__":
    # 获取 S&P 500 股票列表
    download_and_save_sp500_data()
    # cache the data
    data = pd.read_parquet("data/sp500_data.parquet")

    # data_df = get_metrics("GOOG", "2024-01-01", "2024-05-13")
