import backtrader as bt
import pandas as pd
import matplotlib.pyplot as plt
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

        # Log for storing cash balance over time
        self.cash_log = []  # To store cash value at each step
        self.datetime_log = []  # To store the datetime at each step

    def next(self):
        # Store current cash and date
        self.cash_log.append(self.broker.get_cash())
        self.datetime_log.append(self.data.datetime.datetime(0))  # Capture datetime

        if not self.position:  # If not in the market
            if self.sma_short[0] > self.sma_long[0] and self.rsi[0] < self.params.rsi_lower:
                self.buy()
                print(f"Buy at {self.data.close[0]}")
        elif self.sma_short[0] < self.sma_long[0] or self.rsi[0] > self.params.rsi_upper:
            print(f"Sell at {self.data.close[0]}")
            self.sell()


def run_backtest(df):
    # Initialize Cerebro
    cerebro = bt.Cerebro()

    # Add the custom strategy
    cerebro.addstrategy(CustomStrategy)

    # Convert the DataFrame to a Backtrader data feed
    data = bt.feeds.PandasData(
        dataname=df,
        datetime=None,  # Index is used as the datetime column
        open='Open',  # Column name for the open price
        high='High',  # Column name for the high price
        low='Low',  # Column name for the low price
        close='Close',  # Column name for the close price
        volume='Volume',  # Column name for the volume
        openinterest=-1  # Not used, set to -1
    )

    # Add data feed to Cerebro
    cerebro.adddata(data)

    # Set initial cash
    cerebro.broker.set_cash(10000)


    # Print starting conditions
    print(f"Starting Portfolio Value: {cerebro.broker.getvalue():.2f}")

    # Run the backtest and get the strategy instance
    strategies = cerebro.run(stdstats=False)
    strategy = strategies[0]  # Get the first (and only) strategy instance

    # Print final portfolio value
    print(f"Ending Portfolio Value: {cerebro.broker.getvalue():.2f}")

    # Plot the cash balance over time using matplotlib
    plt.figure(figsize=(10, 6))
    plt.plot(strategy.datetime_log, strategy.cash_log, label="Cash Balance")
    plt.xlabel("Date")
    plt.ylabel("Cash Value")
    plt.title("Cash Balance Over Time")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    symbol = 'GOOG'
    file_path = f"../data_collection/{symbol}_minute_data.csv"  # Replace with your CSV file path

    # Read the CSV file
    df = pd.read_csv(file_path, parse_dates=True, index_col=0)

    # Filter to keep only the last 7 days of data
    # last_week_data = df.loc[df.index >= df.index[-1] - timedelta(days=21)]

    # Run the backtest with the last week's data
    run_backtest(df)
