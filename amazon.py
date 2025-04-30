import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.lite.python.interpreter import Interpreter
from sklearn.preprocessing import MinMaxScaler

# --- Load Dataset ---
@st.cache_data
def load_data():
    df = pd.read_csv("Amazon.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values("Date")
    return df[['Date', 'Close']]

# --- Load TFLite Model ---
@st.cache_resource
def load_model():
    interpreter = Interpreter(model_path="amazon_stock_price.tflite")
    interpreter.allocate_tensors()
    return interpreter

# --- Make Prediction ---
def predict_future(data, n_days, model, look_back=60):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))

    last_sequence = scaled_data[-look_back:]
    predictions = []

    for _ in range(n_days):
        input_data = np.array(last_sequence[-look_back:]).reshape(1, look_back, 1).astype(np.float32)

        input_index = model.get_input_details()[0]['index']
        output_index = model.get_output_details()[0]['index']

        model.set_tensor(input_index, input_data)
        model.invoke()
        pred = model.get_tensor(output_index)[0][0]

        predictions.append(pred)
        last_sequence = np.append(last_sequence, [[pred]], axis=0)

    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

# --- Streamlit UI ---
st.title("📈 Prediksi Harga Saham Amazon (TFLite Model)")

df = load_data()
model = load_model()

st.line_chart(df.set_index("Date")["Close"], use_container_width=True)

n_days = st.number_input("Masukkan jumlah hari prediksi:", min_value=1, max_value=365, value=60)

if st.button("Prediksi"):
    close_data = df["Close"].values
    future_preds = predict_future(close_data, n_days, model)

    last_date = df["Date"].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_days)

    pred_df = pd.DataFrame({"Date": future_dates, "Predicted Price": future_preds})
    pred_df.set_index("Date", inplace=True)

    st.line_chart(pred_df, use_container_width=True)
    st.dataframe(pred_df)

