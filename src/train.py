import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

from data_loader import load_data
from preprocess import preprocess

df = load_data()
df = preprocess(df)

df['Prev'] = df['Close'].shift(1)
df = df.dropna()

X = df[['Prev']]
y = df['Close']

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, "models/model.pkl")

print("Model trained")
