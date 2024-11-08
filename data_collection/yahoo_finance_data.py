from datetime import timedelta, datetime

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


class StockAnalyzer:
    def __init__(self):
        self.history_data = None

    def fetch_history(self, ticker_symbol, period='1y', interval='1d'):
        """Fetches historical stock data and summary data."""
        stock = yf.Ticker(ticker_symbol)
        self.history_data = stock.history(period=period, interval=interval)

        # Add stock summary info like P/E ratio
        info = stock.info
        self.history_data['PE_Ratio'] = info.get('trailingPE', np.nan)  # Adding P/E ratio

        return self.history_data

    def plot_stock_data(self, ticker_symbol):
        """Plots the stock's closing price, 20-day SMA, and 50-day SMA."""
        if self.history_data is None:
            self.fetch_history(ticker_symbol)

        # Ensure indicators are calculated before plotting
        self.add_indicators()
        # Plotting
        plt.figure(figsize=(14, 7))

        # Plot Closing Price
        plt.plot(self.history_data.index, self.history_data['Close'], label='Closing Price', color='blue')

        # Plot 20-day and 50-day SMAs
        plt.plot(self.history_data.index, self.history_data['SMA_20'], label='20-day SMA', color='red', linestyle='--')
        plt.plot(self.history_data.index, self.history_data['SMA_50'], label='50-day SMA', color='green',
                 linestyle='--')

        # Customize the plot
        plt.title(f"{ticker_symbol} Stock Price")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.legend()
        plt.grid(True)
        plt.show()

    def fetch_minute_data(self, ticker_symbol, start_date, end_date):
        """Fetches historical 1-minute interval data over a period, handling Yahoo Finance's 7-day limit."""
        all_data = []
        current_start = start_date

        while current_start < end_date:
            current_end = min(current_start + timedelta(days=7), end_date)
            print(f"Downloading data from {current_start} to {current_end}")
            data = yf.download(
                ticker_symbol,
                start=current_start.strftime('%Y-%m-%d'),
                end=current_end.strftime('%Y-%m-%d'),
                interval='1m'
            )
            all_data.append(data)
            current_start = current_end

        # Concatenate all dataframes
        full_data = pd.concat(all_data)
        full_data = full_data[~full_data.index.duplicated(keep='first')]
        self.history_data = full_data  # Store in the instance for further analysis
        self.history_data.columns = [col[0] for col in analyzer.history_data.columns]

        return full_data

    def add_indicators(self):
        """Adds commonly used indicators like SMA and RSI to the data."""
        if self.history_data is None:
            raise ValueError("Historical data not available. Fetch data first.")

        # Add SMA indicators
        self.calculate_moving_average(window=20)
        self.calculate_moving_average(window=50)

        # Add RSI indicator
        self.calculate_rsi(window=14)

    @staticmethod
    def get_summary_info(ticker_symbol):
        """Returns a summary of the stock's basic info."""
        stock = yf.Ticker(ticker_symbol)
        return stock.info

    def calculate_moving_average(self, window=20):
        """Calculates the moving average for the specified window size."""
        if self.history_data is None:
            raise ValueError("Historical data not available. Fetch history first.")
        self.history_data[f'SMA_{window}'] = self.history_data['Close'].rolling(window=window).mean()
        return self.history_data[[f'SMA_{window}']]

    def calculate_rsi(self, window=14):
        """Calculates the Relative Strength Index (RSI) for the specified window size."""
        delta = self.history_data['Close'].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

        rs = gain / loss
        self.history_data['RSI'] = 100 - (100 / (1 + rs))
        return self.history_data['RSI']

    def get_max_min_price(self):
        """Finds the maximum and minimum closing prices in the fetched history."""
        if self.history_data is None:
            raise ValueError("Historical data not available. Fetch history first.")
        max_price = self.history_data['Close'].max()
        min_price = self.history_data['Close'].min()
        return {'Max Price': max_price, 'Min Price': min_price}

    @staticmethod
    def get_day_range(ticker_symbol):
        """Returns the day's high and low price."""
        stock = yf.Ticker(ticker_symbol)
        data = stock.history(period='1d')
        return {'Day Low': data['Low'].iloc[-1], 'Day High': data['High'].iloc[-1]}

    @staticmethod
    def get_52_week_range(ticker_symbol):
        """Returns the 52-week high and low price."""
        stock = yf.Ticker(ticker_symbol)
        return {'52 Week Low': stock.info.get('fiftyTwoWeekLow'), '52 Week High': stock.info.get('fiftyTwoWeekHigh')}

    @staticmethod
    def get_pe_ratios(ticker_symbol):
        """Returns all variations of the P/E ratio."""
        stock = yf.Ticker(ticker_symbol)
        return {
            'Trailing P/E': stock.info.get('trailingPE'),
            'Forward P/E': stock.info.get('forwardPE')
        }

    @staticmethod
    def get_volume_info(ticker_symbol):
        """Returns the current day's volume and the average volume over the last 3 months."""
        stock = yf.Ticker(ticker_symbol)
        return {
            'Volume': stock.info.get('volume'),
            'Average Volume (3 months)': stock.info.get('averageVolume')
        }

    @staticmethod
    def get_market_cap(ticker_symbol):
        """Returns the market capitalization."""
        stock = yf.Ticker(ticker_symbol)
        return stock.info.get('marketCap')

    @staticmethod
    def get_dividend_dates(ticker_symbol):
        """Returns the dividend and ex-dividend dates."""
        stock = yf.Ticker(ticker_symbol)
        return {
            'Dividend Date': stock.info.get('dividendDate'),
            'Ex-Dividend Date': stock.info.get('exDividendDate')
        }

    @staticmethod
    def get_income_statement_info(ticker_symbol):
        """Returns income statement metrics."""
        stock = yf.Ticker(ticker_symbol)
        return {
            'Revenue': stock.financials.loc['Total Revenue'].iloc[0],
            'Revenue Per Share': stock.info.get('revenuePerShare'),
            'Quarterly Revenue Growth': stock.info.get('quarterlyRevenueGrowth')
        }

    @staticmethod
    def get_balance_sheet_info(ticker_symbol):
        """Returns balance sheet metrics."""
        stock = yf.Ticker(ticker_symbol)
        return {
            'Total Cash Per Share': stock.info.get('totalCashPerShare'),
            'Total Debt': stock.balance_sheet.loc['Total Debt'].iloc[0],
            'Debt/Equity': stock.info.get('debtToEquity'),
            'Book Value Per Share': stock.info.get('bookValue')
        }

    @staticmethod
    def get_free_cash_flow(ticker_symbol):
        """Returns the free cash flow."""
        stock = yf.Ticker(ticker_symbol)
        return stock.cashflow.loc['Free Cash Flow'].iloc[0]

    def prepare_features(self):
        """Prepare features and target variable for modeling."""
        if self.history_data is None:
            raise ValueError("Historical data not available. Fetch history first.")

        # Calculate additional indicators
        self.calculate_moving_average(window=20)  # 20-day SMA
        self.calculate_moving_average(window=50)  # 50-day SMA
        self.calculate_rsi(window=14)  # 14-day RSI

        # Drop NaN values that result from indicator calculations
        self.history_data.dropna(inplace=True)

        # Select features for regression
        X = self.history_data[['SMA_20', 'SMA_50', 'RSI', 'PE_Ratio', 'Volume']]
        y = self.history_data['Close']  # Target variable

        return X, y


# Example usage:
if __name__ == "__main__":
    symbol = 'GOOG'  # Example stock ticker
    analyzer = StockAnalyzer()
    # Fetching minute data for the last 30 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    analyzer.fetch_minute_data(symbol, start_date, end_date)

    # Add indicators
    analyzer.add_indicators()

    # Save minute data to an Excel file for record-keeping
    analyzer.history_data.to_csv(f"{symbol}_minute_data.csv")
    print(f"Data saved to {symbol}_minute_data.csv")

    # Plot the data if needed
    analyzer.plot_stock_data(symbol)

    # Test functions
    # # print(analyzer.get_day_range(symbol))
    # # print(analyzer.get_52_week_range(symbol))
    # # print(analyzer.get_pe_ratios(symbol))
    # # print(analyzer.get_volume_info(symbol))
    # # print(analyzer.get_market_cap(symbol))
    # # print(analyzer.get_dividend_dates(symbol))
    # # print(analyzer.get_income_statement_info(symbol))
    # # print(analyzer.get_balance_sheet_info(symbol))
    # # print(analyzer.get_free_cash_flow(symbol))
