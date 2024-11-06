from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from data_collection.yahoo_finance_data import StockAnalyzer


class StockPredictor:
    def __init__(self):
        self.model = LinearRegression()
        self.X_test = None
        self.y_test = None

    def train(self, X, y):
        """Trains the linear regression model."""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # Fit the model
        self.model.fit(X_train, y_train)

        # Store test data for evaluation
        self.X_test, self.y_test = X_test, y_test

    def predict(self, X):
        """Predicts using the trained model."""
        return self.model.predict(X)


# Example usage
if __name__ == "__main__":
    ticker_symbol = 'goog'
    stock_analyzer = StockAnalyzer()
    stock_analyzer.fetch_history(ticker_symbol)
    X, y = stock_analyzer.prepare_features()

    stock_predictor = StockPredictor()
    stock_predictor.train(X, y)
    last_day_info = X.iloc[[-1]]  # Extract the last row of feature data
    prediction = stock_predictor.predict(last_day_info)
    print(f"Predicted closing price for {ticker_symbol}: {prediction[0]}")
