# sma_ema_backtest_fixed.py
# Usage: python sma_ema_backtest_fixed.py
# Make sure your venv is active and libraries installed:
# pip install pandas numpy matplotlib yfinance

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime

# ---------- USER PARAMETERS ----------
TICKER = "AAPL"                 # change to any ticker e.g. "TSLA"
START = "2020-01-01"
END = datetime.today().strftime("%Y-%m-%d")
SHORT = 50                      # short moving average window
LONG = 200                      # long moving average window
INITIAL_CAPITAL = 100000.0      # starting capital for backtest (USD)
SLIPPAGE_PCT = 0.0005           # e.g., 0.05% slippage per trade
COMMISSION = 0.0                # flat commission per trade (set >0 if you want)
# -------------------------------------

# 0) Safe style load (won't crash if style not present)
def safe_style(name_prefer="seaborn-whitegrid"):
    try:
        plt.style.use(name_prefer)
    except OSError:
        # fallback chain
        for alt in ("seaborn", "ggplot", "bmh", "default"):
            try:
                plt.style.use(alt)
                break
            except Exception:
                continue

safe_style()

# 1) Download price data
print(f"Downloading {TICKER} data from {START} to {END} ...")
df = yf.download(TICKER, start=START, end=END, progress=False)
if df.empty:
    raise SystemExit("Error: no data downloaded. Check ticker or internet connection.")

df = df[['Close']].copy()
df.dropna(inplace=True)

# 2) Compute moving averages
df[f'SMA_{SHORT}'] = df['Close'].rolling(window=SHORT, min_periods=1).mean()
df[f'SMA_{LONG}'] = df['Close'].rolling(window=LONG, min_periods=1).mean()
df[f'EMA_{SHORT}'] = df['Close'].ewm(span=SHORT, adjust=False).mean()
df[f'EMA_{LONG}'] = df['Close'].ewm(span=LONG, adjust=False).mean()

# 3) Generate signals
# signal_raw = 1 when short SMA > long SMA (in-market), else 0 (out)
df['signal_raw'] = 0
df.loc[df[f'SMA_{SHORT}'] > df[f'SMA_{LONG}'], 'signal_raw'] = 1
# signal shows changes: +1 = buy event, -1 = sell event
df['signal'] = df['signal_raw'].diff().fillna(0)

# 4) Plot price and SMAs and mark buy/sell events
fig, ax = plt.subplots(figsize=(14,6))
ax.plot(df.index, df['Close'], label=f'{TICKER} Close', alpha=0.6)
ax.plot(df.index, df[f'SMA_{SHORT}'], label=f'SMA {SHORT}', linestyle='--')
ax.plot(df.index, df[f'SMA_{LONG}'], label=f'SMA {LONG}', linestyle='--')

buys = df[df['signal'] == 1]
sells = df[df['signal'] == -1]
ax.scatter(buys.index, buys['Close'], marker='^', color='g', s=70, label='Buy')
ax.scatter(sells.index, sells['Close'], marker='v', color='r', s=70, label='Sell')

ax.set_title(f"{TICKER} Price with SMA {SHORT}/{LONG} Crossovers")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()

# 5) Backtest logic (simple full allocation when signal_raw == 1)
df['position'] = df['signal_raw'].astype(int)  # 1 if in-market, 0 otherwise
df['market_return'] = df['Close'].pct_change().fillna(0)
# strategy uses previous day's position to avoid look-ahead: enter at next bar
df['strategy_return'] = df['position'].shift(1).fillna(0) * df['market_return']

# 6) Trading costs (approx): apply when trade occurs (position changes)
df['trade'] = df['position'].diff().abs().fillna(0)  # 1 on enter or exit
# commission expressed as fraction of trade value: commission / price (approx)
df['trade_cost_pct'] = df['trade'] * (SLIPPAGE_PCT + (COMMISSION / df['Close'].replace(0, np.nan)))
df['strategy_return_net'] = df['strategy_return'] - df['trade_cost_pct'].fillna(0)

# 7) Equity curves
df['strategy_equity'] = (1 + df['strategy_return_net']).cumprod() * INITIAL_CAPITAL
df['market_equity'] = (1 + df['market_return']).cumprod() * INITIAL_CAPITAL

# 8) Performance metrics
def calc_perf(equity_series):
    days = (equity_series.index[-1] - equity_series.index[0]).days
    years = days / 365.25 if days > 0 else np.nan
    total_ret = equity_series.iloc[-1] / equity_series.iloc[0] - 1
    cagr = (1 + total_ret) ** (1/years) - 1 if years and years > 0 else np.nan
    # Convert equity to daily returns for volatility/sharpe
    daily_rets = equity_series.pct_change().dropna()
    ann_vol = daily_rets.std() * np.sqrt(252) if not daily_rets.empty else np.nan
    ann_return = daily_rets.mean() * 252 if not daily_rets.empty else np.nan
    sharpe = (ann_return / ann_vol) if (ann_vol and ann_vol != 0) else np.nan
    roll_max = equity_series.cummax()
    drawdown = (equity_series - roll_max) / roll_max
    max_dd = drawdown.min()
    return {
        'Total Return': total_ret,
        'CAGR': cagr,
        'Annual Volatility': ann_vol,
        'Annual Return (ann.)': ann_return,
        'Sharpe': sharpe,
        'Max Drawdown': max_dd
    }

perf_strategy = calc_perf(df['strategy_equity'])
perf_market = calc_perf(df['market_equity'])

print("\n=== Performance Summary ===")
print("Strategy (SMA crossover):")
for k,v in perf_strategy.items():
    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
print("\nBuy-and-Hold (market):")
for k,v in perf_market.items():
    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

# 9) Plot equity curves
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(df.index, df['market_equity'], label='Buy & Hold', linewidth=1)
ax.plot(df.index, df['strategy_equity'], label='SMA Strategy', linewidth=1)
ax.set_ylabel("Portfolio Value")
ax.set_title("Equity Curve: Strategy vs Buy & Hold")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()

# 10) (Optional) print last rows for inspection
print("\nLast 6 rows (Close, SMA short/long, position, strategy equity):")
print(df[['Close', f'SMA_{SHORT}', f'SMA_{LONG}', 'position', 'strategy_equity']].tail(6))

