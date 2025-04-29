import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from statsmodels.tsa.arima.model import ARIMA

# アプリタイトル
st.title("📈 株価予測アプリ")

# 銘柄コードを入力
ticker = st.text_input("株の銘柄コードを入力してください（例: AAPL や 7203.T）", "AAPL")

# 期間選択
period = st.selectbox(
    "表示する期間を選んでください",
    ("1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"),
    index=3  # デフォルトは"1y"
)

# データ取得
data = yf.download(ticker, period=period)

# 予測用データの準備
data['Date'] = pd.to_datetime(data.index)
data['Date_num'] = data['Date'].map(lambda x: x.toordinal())

# 目的変数と特徴量を分ける
X = data[['Date_num']]
y = data['Close']

# モデル準備
lr_model = LinearRegression()
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
svm_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
arima_model = ARIMA(y, order=(5, 1, 0))

# 学習
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
lr_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
svm_model.fit(X_train, y_train)
arima_model_fit = arima_model.fit()

# ユーザーが予測する期間を指定
forecast_days = st.number_input(
    "予測する期間（日数）を入力してください（例: 30日、60日）",
    min_value=1,
    max_value=365,  # 最大365日（1年分）
    value=30,  # デフォルトは30日
    step=1
)

# 未来予測
future_dates = pd.date_range(start=data.index[-1], periods=forecast_days + 1, freq='D')[1:]  # 予測期間の日付
future_dates_num = future_dates.map(lambda x: x.toordinal()).values.reshape(-1, 1)

lr_predictions = lr_model.predict(future_dates_num)
rf_predictions = rf_model.predict(future_dates_num)
svm_predictions = svm_model.predict(future_dates_num)
arima_predictions = arima_model_fit.forecast(steps=forecast_days)

# 移動平均線など
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

# チェックボックスと説明
st.sidebar.title("表示設定")

show_actual = st.sidebar.checkbox('実際の株価', value=True)
if show_actual:
    st.sidebar.caption("実際の過去の株価推移を表示します。")

show_lr_prediction = st.sidebar.checkbox('予測株価（線形回帰）', value=True)
if show_lr_prediction:
    st.sidebar.caption("過去の傾向を直線的に外挿したシンプルな予測です。")

show_rf_prediction = st.sidebar.checkbox('予測株価（ランダムフォレスト）', value=True)
if show_rf_prediction:
    st.sidebar.caption("決定木の集合体による高精度な予測手法です。")

show_svm_prediction = st.sidebar.checkbox('予測株価（SVM）', value=True)
if show_svm_prediction:
    st.sidebar.caption("データのパターンをとらえて予測する機械学習モデルです。")

show_arima_prediction = st.sidebar.checkbox('予測株価（ARIMA）', value=True)
if show_arima_prediction:
    st.sidebar.caption("時系列データ専用の統計的予測モデルです。")

show_SMA = st.sidebar.checkbox('SMA（単純移動平均）', value=True)
if show_SMA:
    st.sidebar.caption("過去n日間の平均を単純にとったラインです。")

show_EMA = st.sidebar.checkbox('EMA（指数移動平均）', value=True)
if show_EMA:
    st.sidebar.caption("最近のデータを重視する指数加重型の移動平均線です。")

show_BB = st.sidebar.checkbox('ボリンジャーバンド', value=True)
if show_BB:
    st.sidebar.caption("株価の変動範囲を示すバンド（±2σ）を表示します。")

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

st.header("📊 分析結果")

# 直近5日間の移動平均線の傾き（上昇 or 下降）
if data['SMA25'].iloc[-1] > data['SMA25'].iloc[-5]:
    trend = "上昇トレンドです📈"
else:
    trend = "下降トレンドです📉"

st.write(f"現在のトレンドは：**{trend}**")

# 最新の実際の株価とランダムフォレスト予測の最後を取得
latest_actual = y.iloc[-1]  # yの最後の実際の株価
latest_rf_prediction = rf_predictions[-1]  # rf_predictionsの最後の予測値

# もしlatest_rf_predictionがSeriesであればスカラー値に変換
if isinstance(latest_rf_prediction, pd.Series):
    latest_rf_prediction = latest_rf_prediction.item()

# もしlatest_actualがSeriesであればスカラー値に変換
if isinstance(latest_actual, pd.Series):
    latest_actual = latest_actual.item()

# 最新の値を比較
if latest_rf_prediction > latest_actual:
    forecast_message = "今後株価は上昇する可能性があります。"
else:
    forecast_message = "今後株価は下降する可能性があります。"

st.write(f"ランダムフォレストによる予測では、**{forecast_message}**")

# 最新のBB上限と下限を取得
bb_upper = data['BB_upper'].iloc[-1]
bb_lower = data['BB_lower'].iloc[-1]

# Seriesなら数値にする（安全）
bb_upper = bb_upper.item() if isinstance(bb_upper, pd.Series) else bb_upper
bb_lower = bb_lower.item() if isinstance(bb_lower, pd.Series) else bb_lower

# ボリンジャーバンド幅
bb_width = bb_upper - bb_lower

# 株価平均もfloatにしておく
close_mean = data['Close'].mean()

# close_meanがSeriesであればスカラーに変換
close_mean = close_mean.item() if isinstance(close_mean, pd.Series) else close_mean

# 比較（ここで必ず float > float になる）
if bb_width > (close_mean * 0.1):
    volatility = "ボラティリティが高まっています（値動きが激しい）⚡"
else:
    volatility = "ボラティリティは低い状態です（安定している）🛌"

st.write(f"現在のボリンジャーバンド分析：**{volatility}**")

# 直近の終値を取得
last_close = data['Close'].iloc[-1]

# 未来データ作成（全部同じ値スタート）
future_df = pd.DataFrame({
    'Close': [last_close] * forecast_days
})

# インデックスは未来日付
future_df.index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq='B')

# 未来予測
future_predictions = rf_model.predict(future_df)

# DataFrameにまとめる
predicted_future = pd.DataFrame({
    'Date': future_df.index,
    'Predicted_Close': future_predictions
}).set_index('Date')

# グラフと表で出力
st.subheader(f"未来{forecast_days}日間の株価予測📈")
st.line_chart(predicted_future['Predicted_Close'])
st.dataframe(predicted_future)
