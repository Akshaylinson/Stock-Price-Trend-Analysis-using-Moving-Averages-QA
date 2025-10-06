"""
sma_ema_final.py

Final patched SMA/EMA crossover backtest script.

Requirements:
    pip install pandas numpy matplotlib yfinance

Run:
    python sma_ema_final.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
import math
import sys

# ---------- USER PARAMETERS ----------
TICKER = "AAPL"                 # change to your favorite ticker
START = "2020-01-01"
END = datetime.today().strftime("%Y-%m-%d")
SHORT = 50                      # short MA window (days)
LONG = 200                      # long MA window (days)
INITIAL_CAPITAL = 100000.0      # starting capital (currency)
ALLOC_PCT = 0.10                # allocate 10% of current equity when entering
SLIPPAGE_PCT = 0.0005           # 0.05% slippage per trade (fraction)
COMMISSION = 1.0                # flat commission per trade (currency units)
RISK_FREE = 0.0                 # assume 0 for Sharpe calc
# -------------------------------------

def safe_style(name_prefer="seaborn-whitegrid"):
    try:
        plt.style.use(name_prefer)
    except OSError:
        for alt in ("seaborn", "ggplot", "bmh", "default"):
            try:
                plt.style.use(alt)
                break
            except Exception:
                continue

def download_data(ticker):
    # Use auto_adjust to handle corporate actions cleanly
    df = yf.download(ticker, start=START, end=END, progress=False, auto_adjust=True)
    return df

def normalize_columns(df):
    """
    Flatten yfinance MultiIndex columns into simple names like 'Open','High','Low','Close','Volume'.
    Works for single-ticker and multi-ticker responses.
    """
    if isinstance(df.columns, pd.MultiIndex):
        # If first level contains Open/Close etc., keep that
        first_levels = df.columns.get_level_values(0)
        if any(item in ('Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close') for item in first_levels):
            df.columns = first_levels
        else:
            # fallback: join multiindex tuples into single names
            df.columns = ['_'.join([str(c) for c in col if c is not None]) for col in df.columns]
    return df

def safe_int_shares(x):
    try:
        return int(math.floor(x))
    except Exception:
        return 0

def run_sim(position_signal, df_local):
    cash = INITIAL_CAPITAL
    shares = 0
    equity_list = []
    trades = []
    prev_pos = 0

    for date, row in df_local.iterrows():
        price_open = float(row['Open'])
        price_close = float(row['Close'])
        target_pos = int(position_signal.loc[date])

        # Enter (flat -> long)
        if prev_pos == 0 and target_pos == 1:
            current_equity = cash + shares * price_open
            allocate = current_equity * ALLOC_PCT
            exec_price = price_open * (1 + SLIPPAGE_PCT)
            if allocate - COMMISSION > exec_price:
                n_shares = safe_int_shares((allocate - COMMISSION) / exec_price)
            else:
                n_shares = 0
            if n_shares > 0:
                trade_value = n_shares * exec_price
                fees = COMMISSION
                cash -= (trade_value + fees)
                shares += n_shares
                trades.append({'date': date, 'side': 'BUY', 'price': exec_price,
                               'shares': n_shares, 'trade_value': trade_value, 'fees': fees})

        # Exit (long -> flat)
        elif prev_pos == 1 and target_pos == 0:
            if shares > 0:
                exec_price = price_open * (1 - SLIPPAGE_PCT)
                trade_value = shares * exec_price
                fees = COMMISSION
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

def perf_stats(equity_series):
    if isinstance(equity_series, pd.DataFrame):
        equity_series = equity_series.squeeze()
    equity_series = pd.to_numeric(equity_series, errors='coerce').dropna()

    if equity_series.empty:
        return {
            'Total Return': np.nan,
            'CAGR': np.nan,
            'Annual Volatility': np.nan,
            'Annual Return (ann.)': np.nan,
            'Sharpe': np.nan,
            'Sortino': np.nan,
            'Max Drawdown': np.nan,
            'MAR': np.nan
        }

    total_ret = equity_series.iloc[-1] / equity_series.iloc[0] - 1
    days = (equity_series.index[-1] - equity_series.index[0]).days
    years = days / 365.25 if days > 0 else np.nan
    cagr = (1 + total_ret) ** (1/years) - 1 if years and years > 0 else np.nan

    daily_rets = equity_series.pct_change().dropna()
    daily_rets = pd.to_numeric(daily_rets, errors='coerce').dropna()

    ann_vol = daily_rets.std() * np.sqrt(252) if not daily_rets.empty else np.nan
    ann_return = daily_rets.mean() * 252 if not daily_rets.empty else np.nan
    sharpe = (ann_return - RISK_FREE) / ann_vol if ann_vol and ann_vol != 0 else np.nan

    neg_rets = daily_rets[daily_rets < 0]
    downside_vol = neg_rets.std() * np.sqrt(252) if not neg_rets.empty else np.nan
    sortino = (ann_return - RISK_FREE) / downside_vol if downside_vol and downside_vol != 0 else np.nan

    roll_max = equity_series.cummax()
    drawdown = (equity_series - roll_max) / roll_max
    max_dd = drawdown.min()
    mar = cagr / abs(max_dd) if max_dd and max_dd != 0 else np.nan

    return {
        'Total Return': total_ret,
        'CAGR': cagr,
        'Annual Volatility': ann_vol,
        'Annual Return (ann.)': ann_return,
        'Sharpe': sharpe,
        'Sortino': sortino,
        'Max Drawdown': max_dd,
        'MAR': mar
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
    wins = pnl_arr[pnl_arr > 0]
    losses = pnl_arr[pnl_arr <= 0]
    win_rate = len(wins) / len(pnl_arr)
    avg_win = wins.mean() if wins.size > 0 else np.nan
    avg_loss = losses.mean() if losses.size > 0 else np.nan

    return {
        'Trade Count': len(pnl_arr),
        'Win Rate': win_rate,
        'Avg Win (pct)': avg_win,
        'Avg Loss (pct)': avg_loss
    }

def print_perf(name, perf, tstats, trades_df, equity_series):
    print(f"\n=== {name} PERFORMANCE ===")
    for k, v in perf.items():
        print(f"{k:20}: {v:.6f}" if isinstance(v, float) else f"{k:20}: {v}")
    print("Trade stats:")
    for k, v in tstats.items():
        print(f"  {k:15}: {v:.6f}" if isinstance(v, float) else f"  {k:15}: {v}")
    if trades_df.empty:
        print("  WARNING: No trades were generated for this strategy.")
    if equity_series.empty:
        print("  WARNING: Equity series is empty (no data to compute performance).")

def main():
    safe_style()
    # download and normalize
    df_raw = download_data(TICKER)
    if df_raw.empty:
        sys.exit("No data downloaded. Check ticker and internet connection.")
    df_norm = normalize_columns(df_raw.copy())
    if 'Open' not in df_norm.columns or 'Close' not in df_norm.columns:
        sys.exit("ERROR: Required columns 'Open' and 'Close' not found after normalization.")
    df = df_norm[['Open', 'Close']].dropna()

    # indicators and signals
    df[f'SMA_{SHORT}'] = df['Close'].rolling(SHORT, min_periods=1).mean()
    df[f'SMA_{LONG}']  = df['Close'].rolling(LONG, min_periods=1).mean()
    df[f'EMA_{SHORT}'] = df['Close'].ewm(span=SHORT, adjust=False).mean()
    df[f'EMA_{LONG}']  = df['Close'].ewm(span=LONG, adjust=False).mean()
    df['sma_in'] = (df[f'SMA_{SHORT}'] > df[f'SMA_{LONG}']).astype(int)
    df['ema_in'] = (df[f'EMA_{SHORT}'] > df[f'EMA_{LONG}']).astype(int)

    # run sims
    sma_trades, sma_equity = run_sim(df['sma_in'], df)
    ema_trades, ema_equity = run_sim(df['ema_in'], df)

    # stats
    sma_perf = perf_stats(sma_equity)
    ema_perf = perf_stats(ema_equity)
    sma_tstats = trade_stats(sma_trades)
    ema_tstats = trade_stats(ema_trades)

    print_perf("SMA Crossover", sma_perf, sma_tstats, sma_trades, sma_equity)
    print_perf("EMA Crossover", ema_perf, ema_tstats, ema_trades, ema_equity)

    # plot equity curves vs buy & hold
    buy_hold_equity = (1 + df['Close'].pct_change().fillna(0)).cumprod() * INITIAL_CAPITAL
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(sma_equity.index, sma_equity, label='SMA Strategy Equity')
    ax.plot(ema_equity.index, ema_equity, label='EMA Strategy Equity')
    ax.plot(buy_hold_equity.index, buy_hold_equity, label='Buy & Hold', linestyle='--', alpha=0.8)
    ax.set_title(f"Equity Curves ({TICKER})")
    ax.set_ylabel("Portfolio Value")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
