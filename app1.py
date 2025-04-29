import yfinance as yf
import matplotlib.pyplot as plt
import japanize_matplotlib
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler

# アプリタイトル
st.title("📈 株価予測アプリ")

# ユーザー入力
ticker = st.text_input("株の銘柄コードを入力してください（例: AAPL や 7203.T）", "AAPL")
period = st.selectbox("表示する期間を選んでください",
                      ("1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"), index=3)

forecast_days = st.number_input(
    "予測する期間（日数）を入力してください（例: 30日、60日）",
    min_value=1, max_value=365, value=30, step=1
)

# 株価データ取得
data = yf.download(ticker, period=period)

if data.empty:
    st.warning("データが取得できませんでした。銘柄コードや期間を確認してください。")
    st.stop()

data['Date'] = pd.to_datetime(data.index)
data['Date_num'] = data['Date'].map(lambda x: x.toordinal())
X = data[['Date_num']]
y = data['Close']

# データ分割（未来予測用にシャッフルしない）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# モデル学習
lr_model = LinearRegression().fit(X_train, y_train)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
svm_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1).fit(X_train, y_train)
arima_model = ARIMA(y, order=(5, 1, 0)).fit()

# 未来の日付生成
future_dates = pd.date_range(start=data.index[-1], periods=forecast_days + 1, freq='D')[1:]
future_dates_num = future_dates.map(lambda x: x.toordinal()).values.reshape(-1, 1)

# 予測
lr_predictions = lr_model.predict(future_dates_num)
rf_predictions = rf_model.predict(future_dates_num)
svm_predictions = svm_model.predict(future_dates_num)
arima_predictions = arima_model.forecast(steps=forecast_days)

# LSTM予測
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(np.array(y).reshape(-1, 1))

def create_dataset(dataset, time_step=60):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), 0])
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 60
if len(scaled_data) > time_step + 1:
    X_lstm, y_lstm = create_dataset(scaled_data, time_step)
    X_lstm = X_lstm.reshape(X_lstm.shape[0], X_lstm.shape[1], 1)

    lstm_model = Sequential()
    lstm_model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
    lstm_model.add(LSTM(50))
    lstm_model.add(Dense(1))
    lstm_model.compile(loss='mean_squared_error', optimizer='adam')
    lstm_model.fit(X_lstm, y_lstm, epochs=5, batch_size=32, verbose=0)

    lstm_input = scaled_data[-time_step:]
    lstm_predictions = []
    for _ in range(forecast_days):
        input_reshaped = lstm_input.reshape(1, time_step, 1)
        pred = lstm_model.predict(input_reshaped, verbose=0)
        lstm_predictions.append(pred[0][0])
        lstm_input = np.append(lstm_input[1:], pred)

    lstm_predictions = scaler.inverse_transform(np.array(lstm_predictions).reshape(-1, 1)).flatten()
else:
    lstm_predictions = None
    st.warning("データが少ないため、LSTM予測はスキップされました。")

# 移動平均線
data['SMA5'] = data['Close'].rolling(window=5).mean()
data['SMA25'] = data['Close'].rolling(window=25).mean()
data['SMA75'] = data['Close'].rolling(window=75).mean()
data['EMA5'] = data['Close'].ewm(span=5, adjust=False).mean()
data['EMA25'] = data['Close'].ewm(span=25, adjust=False).mean()
data['EMA75'] = data['Close'].ewm(span=75, adjust=False).mean()

# ボリンジャーバンド
sma25 = data['Close'].rolling(window=25).mean()
rolling_std = data['Close'].rolling(window=25).std()
data['BB_upper'] = sma25 + (2 * rolling_std)
data['BB_lower'] = sma25 - (2 * rolling_std)

# サイドバー設定
st.sidebar.title("表示設定")
show_actual = st.sidebar.checkbox('実際の株価', value=True)
show_lr_prediction = st.sidebar.checkbox('予測株価（線形回帰）', value=False)
show_rf_prediction = st.sidebar.checkbox('予測株価（ランダムフォレスト）', value=False)
show_svm_prediction = st.sidebar.checkbox('予測株価（SVM）', value=False)
show_arima_prediction = st.sidebar.checkbox('予測株価（ARIMA）', value=False)
show_lstm_prediction = st.sidebar.checkbox('予測株価（LSTM）', value=False if lstm_predictions is None else False)
show_SMA = st.sidebar.checkbox('SMA（単純移動平均）', value=True)
show_EMA = st.sidebar.checkbox('EMA（指数移動平均）', value=False)
show_BB = st.sidebar.checkbox('ボリンジャーバンド', value=False)

# グラフ描画
fig, ax = plt.subplots(figsize=(14, 7))

if show_actual:
    ax.plot(data['Date'], y, label="実際の株価", color="blue")

if show_lr_prediction:
    ax.plot(future_dates, lr_predictions, label="予測（線形回帰）", color="red", linestyle="--")

if show_rf_prediction:
    ax.plot(future_dates, rf_predictions, label="予測（ランダムフォレスト）", color="green", linestyle="--")

if show_svm_prediction:
    ax.plot(future_dates, svm_predictions, label="予測（SVM）", color="orange", linestyle="--")

if show_arima_prediction:
    ax.plot(future_dates, arima_predictions, label="予測（ARIMA）", color="purple", linestyle="--")

if show_lstm_prediction and lstm_predictions is not None:
    ax.plot(future_dates[:len(lstm_predictions)], lstm_predictions, label="予測（LSTM）", color="pink", linestyle="--")

if show_SMA:
    ax.plot(data['Date'], data['SMA5'], label="SMA 5日", linestyle="--", color="cyan")
    ax.plot(data['Date'], data['SMA25'], label="SMA 25日", linestyle="--", color="magenta")
    ax.plot(data['Date'], data['SMA75'], label="SMA 75日", linestyle="--", color="yellow")

if show_EMA:
    ax.plot(data['Date'], data['EMA5'], label="EMA 5日", linestyle="--", color="orange")
    ax.plot(data['Date'], data['EMA25'], label="EMA 25日", linestyle="--", color="purple")
    ax.plot(data['Date'], data['EMA75'], label="EMA 75日", linestyle="--", color="pink")

if show_BB:
    ax.plot(data['Date'], data['BB_upper'], label="ボリンジャーバンド上限", linestyle="--", color="brown")
    ax.plot(data['Date'], data['BB_lower'], label="ボリンジャーバンド下限", linestyle="--", color="grey")

ax.set_title(f"{ticker} の株価予測と移動平均")
ax.set_xlabel("日付")
ax.set_ylabel("株価（USD）")
ax.legend(loc='upper left', fontsize=10)
ax.grid(True)

st.pyplot(fig)

# 結果の要約
st.header("📊 分析結果")

# トレンドの判定
if len(data['SMA25']) >= 5 and data['SMA25'].iloc[-1] > data['SMA25'].iloc[-5]:
    trend = "上昇トレンドです📈"
else:
    trend = "下降トレンドです📉"

st.write(f"現在のトレンドは：**{trend}**")

# 最新の株価と予測の比較
latest_actual = y.iloc[-1].item()  # floatに変換
latest_rf_prediction = rf_predictions[-1]
predicted_date = future_dates[-1].strftime('%Y-%m-%d')  # 予測対象の日付

st.write(f"現在の株価: ${latest_actual:.2f}")
st.write(f"{forecast_days}日後（{predicted_date}）のランダムフォレスト予測値: ${latest_rf_prediction:.2f}")



if latest_rf_prediction > latest_actual:
    st.write("今後株価は上昇する可能性があります。")
else:
    st.write("今後株価は下降する可能性があります。")

# 予測手法の説明
st.header("🧠 予測手法の説明")

with st.expander("線形回帰（Linear Regression）とは？"):
    st.markdown("""
    線形回帰は、株価の変動を日付との直線関係としてとらえるシンプルな予測手法です。
    長期的なトレンドを把握するのに適していますが、短期的な変動には弱いことがあります。
    """)

with st.expander("ランダムフォレスト（Random Forest）とは？"):
    st.markdown("""
    複数の決定木を組み合わせて予測する手法で、ノイズに強く、非線形な関係を扱えます。
    価格変動が複雑な銘柄にも対応しやすいのが特徴です。
    """)

with st.expander("SVM（サポートベクターマシン）とは？"):
    st.markdown("""
    データの傾向を境界で分類する手法で、非線形な予測が可能です。
    適切なパラメータ設定により、精度が高くなることもあります。
    """)

with st.expander("ARIMAモデルとは？"):
    st.markdown("""
    過去の値動きから将来の株価を予測する時系列モデルです。
    季節性やトレンドを含むデータの予測に強みがあります。
    """)

with st.expander("LSTM（長短期記憶）とは？"):
    st.markdown("""
    過去の時間的なデータの関係性を学習する深層学習モデルで、連続するデータの予測に特に強いです。
    より長期的なパターンを記憶しながら、未来の株価を予測します。
    """)

with st.expander("移動平均とボリンジャーバンドについて"):
    st.markdown("""
    - **SMA（単純移動平均）**：一定期間の平均値を滑らかにしたものです。
    - **EMA（指数移動平均）**：最近の価格に重みをおいて平均化したものです。
    - **ボリンジャーバンド**：価格の標準偏差を利用して、価格の変動範囲を示します。価格が上下限に近づくと反発のサインともされます。
    """)
