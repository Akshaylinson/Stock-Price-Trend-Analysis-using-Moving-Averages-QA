# app_fixed.py
"""
Robust Streamlit backtest UI (fixed).
Run:
    streamlit run app_fixed.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta
import math

st.set_page_config(layout="wide", page_title="SMA/EMA Backtest")

# ----------------- Helper functions -----------------
@st.cache_data(ttl=3600)
def get_data_cached(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Download and normalize data. Cached to avoid repeated downloads."""
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        first_levels = df.columns.get_level_values(0)
        if any(i in ('Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close') for i in first_levels):
            df.columns = first_levels
        else:
            df.columns = ['_'.join([str(c) for c in col if c is not None]) for col in df.columns]
    return df

def safe_int_shares(x):
    try:
        return int(math.floor(x))
    except Exception:
        return 0

def run_sim(position_signal, df_local, alloc_pct, slippage_pct, commission, initial_capital):
    cash = initial_capital
    shares = 0
    equity_list = []
    trades = []
    prev_pos = 0

    for date, row in df_local.iterrows():
        try:
            price_open = float(row['Open'])
            price_close = float(row['Close'])
        except Exception:
            # If open/close missing for a row, skip marking equity (shouldn't normally happen)
            equity_list.append(cash + shares * (row.get('Close', 0) or 0))
            continue

        target_pos = int(position_signal.loc[date])

        # Enter
        if prev_pos == 0 and target_pos == 1:
            current_equity = cash + shares * price_open
            allocate = current_equity * alloc_pct
            exec_price = price_open * (1 + slippage_pct)
            if allocate - commission > exec_price:
                n_shares = safe_int_shares((allocate - commission) / exec_price)
            else:
                n_shares = 0
            if n_shares > 0:
                trade_value = n_shares * exec_price
                fees = commission
                cash -= (trade_value + fees)
                shares += n_shares
                trades.append({'date': date, 'side': 'BUY', 'price': exec_price,
                               'shares': n_shares, 'trade_value': trade_value, 'fees': fees})

        # Exit
        elif prev_pos == 1 and target_pos == 0:
            if shares > 0:
                exec_price = price_open * (1 - slippage_pct)
                trade_value = shares * exec_price
                fees = commission
                cash += (trade_value - fees)
                trades.append({'date': date, 'side': 'SELL', 'price': exec_price,
                               'shares': shares, 'trade_value': trade_value, 'fees': fees})
                shares = 0

        equity = cash + shares * price_close
        equity_list.append(equity)
        prev_pos = target_pos

    equity_series = pd.Series(equity_list, index=df_local.index)
    trades_df = pd.DataFrame(trades)
    return trades_df, equity_series

def perf_stats(equity_series, risk_free=0.0):
    if isinstance(equity_series, pd.DataFrame):
        equity_series = equity_series.squeeze()
    equity_series = pd.to_numeric(equity_series, errors='coerce').dropna()
    if equity_series.empty:
        return {k: np.nan for k in ['Total Return','CAGR','Annual Volatility','Annual Return (ann.)','Sharpe','Sortino','Max Drawdown','MAR']}
    total_ret = equity_series.iloc[-1] / equity_series.iloc[0] - 1
    days = (equity_series.index[-1] - equity_series.index[0]).days
    years = days / 365.25 if days > 0 else np.nan
    cagr = (1 + total_ret) ** (1/years) - 1 if years and years > 0 else np.nan
    daily_rets = equity_series.pct_change().dropna()
    daily_rets = pd.to_numeric(daily_rets, errors='coerce').dropna()
    ann_vol = daily_rets.std() * np.sqrt(252) if not daily_rets.empty else np.nan
    ann_return = daily_rets.mean() * 252 if not daily_rets.empty else np.nan
    sharpe = (ann_return - risk_free) / ann_vol if ann_vol and ann_vol != 0 else np.nan
    neg_rets = daily_rets[daily_rets < 0]
    downside_vol = neg_rets.std() * np.sqrt(252) if not neg_rets.empty else np.nan
    sortino = (ann_return - risk_free) / downside_vol if downside_vol and downside_vol != 0 else np.nan
    roll_max = equity_series.cummax()
    drawdown = (equity_series - roll_max) / roll_max
    max_dd = drawdown.min()
    mar = cagr / abs(max_dd) if max_dd and max_dd != 0 else np.nan
    return {
        'Total Return': total_ret, 'CAGR': cagr, 'Annual Volatility': ann_vol,
        'Annual Return (ann.)': ann_return, 'Sharpe': sharpe, 'Sortino': sortino,
        'Max Drawdown': max_dd, 'MAR': mar
    }

def trade_stats(trades_df):
    if trades_df.empty:
        return {'Trade Count': 0, 'Win Rate': np.nan, 'Avg Win (pct)': np.nan, 'Avg Loss (pct)': np.nan}
    trades = trades_df.reset_index(drop=True)
    pnl_list = []
    last_buy = None
    for _, tr in trades.iterrows():
        if tr['side'] == 'BUY':
            last_buy = tr
        elif tr['side'] == 'SELL' and last_buy is not None:
            buy_total = last_buy['trade_value'] + last_buy['fees']
            sell_total = tr['trade_value'] - tr['fees']
            pnl = sell_total - buy_total
            pnl_pct = pnl / buy_total if buy_total != 0 else np.nan
            pnl_list.append(pnl_pct)
            last_buy = None
    pnl_arr = np.array(pnl_list)
    if pnl_arr.size == 0:
        return {'Trade Count': 0, 'Win Rate': np.nan, 'Avg Win (pct)': np.nan, 'Avg Loss (pct)': np.nan}
    wins = pnl_arr[pnl_arr > 0]; losses = pnl_arr[pnl_arr <= 0]
    return {
        'Trade Count': len(pnl_arr), 'Win Rate': len(wins) / len(pnl_arr),
        'Avg Win (pct)': wins.mean() if wins.size > 0 else np.nan,
        'Avg Loss (pct)': losses.mean() if losses.size > 0 else np.nan
    }

# ----------------- Sidebar controls -----------------
st.sidebar.header("Backtest controls")

ticker = st.sidebar.text_input("Ticker", value="AAPL")

today = datetime.today().date()
three_years_ago = (datetime.today() - timedelta(days=365*3)).date()
start_date = st.sidebar.date_input("Start date", value=three_years_ago)
end_date = st.sidebar.date_input("End date", value=today)
short = st.sidebar.number_input("SHORT (days)", min_value=2, max_value=250, value=50)
long = st.sidebar.number_input("LONG (days)", min_value=5, max_value=500, value=200)
alloc_pct = st.sidebar.slider("Allocation % (per entry)", min_value=0.01, max_value=1.0, value=0.10, step=0.01)
slippage_pct = st.sidebar.slider("Slippage %", min_value=0.0, max_value=0.01, value=0.0005, step=0.0001)
commission = st.sidebar.number_input("Commission (flat)", min_value=0.0, value=1.0, step=0.1)
initial_cap = st.sidebar.number_input("Initial Capital", min_value=100.0, value=100000.0, step=100.0)

st.title("SMA/EMA Backtest Dashboard")
st.markdown("Adjust controls in the left sidebar and click **Run Backtest**.")

# run button
run = st.button("Run Backtest")

# show a small placeholder summary to avoid a blank page
st.write("Choose parameters and press **Run Backtest**. Results will appear below.")

# ----------------- Main action -----------------
if run:
    try:
        df_raw = get_data_cached(ticker, start_date, end_date)
        if df_raw.empty:
            st.error("No data downloaded for that ticker / date range.")
            st.stop()

        if 'Open' not in df_raw.columns or 'Close' not in df_raw.columns:
            st.error("Downloaded data missing Open/Close columns after normalization.")
            st.stop()

        df = df_raw[['Open', 'Close']].dropna()
        # compute indicators
        df[f'SMA_{short}'] = df['Close'].rolling(short, min_periods=1).mean()
        df[f'SMA_{long}'] = df['Close'].rolling(long, min_periods=1).mean()
        df[f'EMA_{short}'] = df['Close'].ewm(span=short, adjust=False).mean()
        df[f'EMA_{long}'] = df['Close'].ewm(span=long, adjust=False).mean()
        df['sma_in'] = (df[f'SMA_{short}'] > df[f'SMA_{long}']).astype(int)
        df['ema_in'] = (df[f'EMA_{short}'] > df[f'EMA_{long}']).astype(int)

        # run sims
        sma_trades, sma_equity = run_sim(df['sma_in'], df, alloc_pct, slippage_pct, commission, initial_cap)
        ema_trades, ema_equity = run_sim(df['ema_in'], df, alloc_pct, slippage_pct, commission, initial_cap)

        # stats
        sma_perf = perf_stats(sma_equity)
        ema_perf = perf_stats(ema_equity)
        sma_tstats = trade_stats(sma_trades)
        ema_tstats = trade_stats(ema_trades)

        # KPI cards
        st.subheader(f"Results for {ticker} (from {start_date} to {end_date})")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Return (SMA)", f"{sma_perf['Total Return']*100:.2f}%" if not pd.isna(sma_perf['Total Return']) else "N/A")
        c2.metric("CAGR (SMA)", f"{sma_perf['CAGR']*100:.2f}%" if not pd.isna(sma_perf['CAGR']) else "N/A")
        c3.metric("Sharpe (SMA)", f"{sma_perf['Sharpe']:.3f}" if not pd.isna(sma_perf['Sharpe']) else "N/A")
        c4.metric("Max Drawdown (SMA)", f"{sma_perf['Max Drawdown']*100:.2f}%" if not pd.isna(sma_perf['Max Drawdown']) else "N/A")

        # equity plot
        buy_hold_equity = (1 + df['Close'].pct_change().fillna(0)).cumprod() * initial_cap
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sma_equity.index, y=sma_equity.values, name='SMA Equity'))
        fig.add_trace(go.Scatter(x=ema_equity.index, y=ema_equity.values, name='EMA Equity'))
        fig.add_trace(go.Scatter(x=buy_hold_equity.index, y=buy_hold_equity.values, name='Buy & Hold', line=dict(dash='dash')))
        fig.update_layout(title="Equity Curves", xaxis_title="Date", yaxis_title="Portfolio Value")
        st.plotly_chart(fig, use_container_width=True)

        # trades table & download
        st.subheader("SMA Trades")
        st.dataframe(sma_trades[['date','side','price','shares','trade_value','fees']].sort_values('date').reset_index(drop=True))
        st.download_button("Download SMA trades CSV", sma_trades.to_csv(index=False).encode('utf-8'), file_name=f"{ticker}_sma_trades.csv")

        st.subheader("EMA Trades")
        st.dataframe(ema_trades[['date','side','price','shares','trade_value','fees']].sort_values('date').reset_index(drop=True))
        st.download_button("Download EMA trades CSV", ema_trades.to_csv(index=False).encode('utf-8'), file_name=f"{ticker}_ema_trades.csv")

        st.subheader("SMA Performance Summary")
        st.write(pd.DataFrame.from_dict(sma_perf, orient='index', columns=['Value']).style.format("{:.6f}"))
        st.subheader("SMA Trade Stats")
        st.write(pd.DataFrame.from_dict(sma_tstats, orient='index', columns=['Value']).style.format("{:.6f}"))

    except Exception as exc:
        st.error("An error occurred while running the backtest. See details below.")
        st.exception(exc)

