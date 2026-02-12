import yfinance as yf

def load_data():
    data = yf.download("USDINR=X", start="2018-01-01")
    return data

if __name__ == "__main__":
    df = load_data()
    print(df.head())
