import joblib
import pandas as pd
import yfinance as yf


def predict_stock_price(symbol: str, period: str = "7d", interval: str = "2m"):
    model_joblib_file = "linear_model.joblib"
    features = ["Date_Ordinal", "Open", "High", "Low", "Volume", "SMA_50", "SMA_200", "RSI_14", "MACD"]

    stock_data = yf.download(symbol, period=period, interval=interval)

    stock_data['Date_Ordinal'] = stock_data.index.map(pd.Timestamp.toordinal)
    stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['SMA_200'] = stock_data['Close'].rolling(window=200).mean()

    def compute_rsi(series, window=14):
        delta = series.diff()

        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    stock_data['RSI_14'] = compute_rsi(stock_data['Close'])

    short_ema = stock_data['Close'].ewm(span=12, adjust=False).mean()
    long_ema = stock_data['Close'].ewm(span=26, adjust=False).mean()
    stock_data['MACD'] = short_ema - long_ema

    latest_data = stock_data.iloc[-1:]
    latest_data = pd.DataFrame(latest_data.values, columns=stock_data.columns)
    actual_close_price = latest_data["Close"].values[0][0]

    stock_data = stock_data[features]
    stock_data.dropna(inplace=True)

    # Loading Trained Linear Regression model
    model = joblib.load(model_joblib_file)

    latest_data = stock_data.iloc[-1:]
    latest_data = pd.DataFrame(latest_data.values, columns=features)
    predicted_price = model.predict(latest_data)

    return predicted_price[0], actual_close_price


predicted_value, actual_close_value = predict_stock_price("MSFT")
print("Actual Value:", actual_close_value)
print("Predicted Value:", predicted_value)
