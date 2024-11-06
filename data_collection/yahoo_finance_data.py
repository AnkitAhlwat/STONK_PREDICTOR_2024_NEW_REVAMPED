import numpy as np
import yfinance as yf


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
    print(analyzer.get_day_range(symbol))
    print(analyzer.get_52_week_range(symbol))
    print(analyzer.get_pe_ratios(symbol))
    print(analyzer.get_volume_info(symbol))
    print(analyzer.get_market_cap(symbol))
    print(analyzer.get_dividend_dates(symbol))
    print(analyzer.get_income_statement_info(symbol))
    print(analyzer.get_balance_sheet_info(symbol))
    print(analyzer.get_free_cash_flow(symbol))
