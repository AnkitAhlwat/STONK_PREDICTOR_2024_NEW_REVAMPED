import backtrader as bt
import pandas as pd
from datetime import timedelta

class CustomStrategy(bt.Strategy):
    params = (
        ('short_sma', 20),
        ('long_sma', 50),
        ('rsi_period', 14),
        ('rsi_upper', 70),
        ('rsi_lower', 30),
    )

    def __init__(self):
        # Add indicators
        self.sma_short = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.short_sma)
        self.sma_long = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.long_sma)
        self.rsi = bt.indicators.RelativeStrengthIndex(self.data.close, period=self.params.rsi_period)

    def next(self):
        # Implement a simple strategy: Buy when short SMA crosses above long SMA and RSI is below 30, sell when it crosses below
        if not self.position:  # If not in the market
            if self.sma_short[0] > self.sma_long[0] and self.rsi[0] < self.params.rsi_lower:
                self.buy()
        elif self.sma_short[0] < self.sma_long[0] or self.rsi[0] > self.params.rsi_upper:
            self.sell()

def run_backtest(df):
    # Initialize Cerebro
    cerebro = bt.Cerebro()

    # Add the custom strategy
    cerebro.addstrategy(CustomStrategy)

    # Convert the DataFrame to a Backtrader data feed
    data = bt.feeds.PandasData(
        dataname=df,
        datetime=None,
        open='Open',
        high='High',
        low='Low',
        close='Close',
        volume='Volume',
        openinterest=-1
    )

    # Add data feed to Cerebro
    cerebro.adddata(data)

    # Set initial cash
    cerebro.broker.set_cash(10000)

    # Set commission
    cerebro.broker.setcommission(commission=0.001)

    # Print starting conditions
    print(f"Starting Portfolio Value: {cerebro.broker.getvalue():.2f}")

    # Run the backtest
    cerebro.run()

    # Print final portfolio value
    print(f"Ending Portfolio Value: {cerebro.broker.getvalue():.2f}")

    # Plot the result
    cerebro.plot()

# Load data from CSV and prepare for backtest
if __name__ == "__main__":
    symbol = 'GOOG'
    file_path = f"../data_collection/{symbol}_minute_data.csv"  # Replace with your CSV file path

    # Read the CSV file
    df = pd.read_csv(file_path, parse_dates=['Datetime'], index_col='Datetime')

    # Filter to keep only the last 7 days of data
    last_week_data = df.loc[df.index >= (df.index.max() - timedelta(days=7))]

    # Run the backtest with the last week's data
    run_backtest(last_week_data)
