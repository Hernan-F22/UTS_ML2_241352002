import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")
st.title("📊 Amazon Stock Price EDA (1997–2020)")

# Upload file CSV
uploaded_file = st.file_uploader("Upload dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    # Baca data
    df = pd.read_csv(uploaded_file)
    
    # Pastikan kolom Date dalam format datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Sidebar: pilih rentang waktu
    st.sidebar.title("Filter Data")
    start_date = st.sidebar.date_input("Start Date", df['Date'].min().date())
    end_date = st.sidebar.date_input("End Date", df['Date'].max().date())
    
    # Filter data berdasarkan tanggal
    mask = (df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))
    df = df.loc[mask]

    st.markdown("## 📈 Close Price Over Time")
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(df['Date'], df['Close'], label='Close Price')
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price (USD)")
    ax1.grid(True)
    ax1.legend()
    st.pyplot(fig1)

    st.markdown("## 📊 Price Distribution (Close)")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    sns.histplot(df['Close'], bins=50, kde=True, ax=ax2)
    ax2.set_xlabel("Close Price")
    st.pyplot(fig2)

    st.markdown("## 🔗 Correlation Matrix")
    corr = df[['Open', 'High', 'Low', 'Close', 'Volume']].corr()
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax3)
    st.pyplot(fig3)

    st.markdown("## 🟡 Moving Averages")
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    fig4, ax4 = plt.subplots(figsize=(12, 5))
    ax4.plot(df['Date'], df['Close'], label='Close', alpha=0.5)
    ax4.plot(df['Date'], df['MA20'], label='MA20', linestyle='--')
    ax4.plot(df['Date'], df['MA50'], label='MA50', linestyle='--')
    ax4.set_xlabel("Date")
    ax4.set_ylabel("Price")
    ax4.legend()
    ax4.grid(True)
    st.pyplot(fig4)

    st.markdown("## 📦 Volume Over Time")
    fig5, ax5 = plt.subplots(figsize=(12, 4))
    ax5.plot(df['Date'], df['Volume'], color='orange')
    ax5.set_xlabel("Date")
    ax5.set_ylabel("Volume")
    st.pyplot(fig5)

else:
    st.info("👈 Silakan unggah file CSV untuk mulai eksplorasi.")