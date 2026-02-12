def preprocess(df):
    df = df[['Close']]
    df = df.dropna()
    return df
