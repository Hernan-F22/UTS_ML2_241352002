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
    # Set indeks ke DateTime
    df.index = pd.to_datetime(df['Date'])

    # Data historis (actual)
    df_past = df[['Close']].copy()
    df_past['Forecast'] = np.nan
    df_past.rename(columns={'Close': 'Actual'}, inplace=True)

    # Jalankan prediksi
    close_data = df_past['Actual'].values
    forecast = predict_future(close_data, n_days, model)

    # Isi nilai prediksi pertama agar menyatu mulus
    df_past.loc[df_past.index[-1], 'Forecast'] = df_past.loc[df_past.index[-1], 'Actual']

    # Buat tanggal untuk prediksi
    future_dates = pd.date_range(start=df_past.index[-1] + pd.Timedelta(days=1), periods=n_days)

    # Buat DataFrame prediksi masa depan
    df_future = pd.DataFrame({
        'Actual': [np.nan] * n_days,
        'Forecast': forecast
    }, index=future_dates)

    # Gabungkan historis dan prediksi
    results = pd.concat([df_past, df_future])

    # Plot dengan matplotlib
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(results.index, results['Actual'], label='Harga Historis')
    ax.plot(results.index, results['Forecast'], label=f'Prediksi {n_days} Hari', color='orange')
    ax.set_title(f'Forecast Saham {n_days} Hari')
    ax.set_xlabel('Tanggal')
    ax.set_ylabel('Harga')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    # Tampilkan di Streamlit
    st.pyplot(fig)

    # Tampilkan tabel hasil forecast
    st.subheader("📋 Hasil Prediksi")
    st.dataframe(df_future.reset_index().rename(columns={'index': 'Date'}))