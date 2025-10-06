.

📊 Stock Price Trend Analysis using Moving Averages

A Python-based Quantitative Research Project that analyzes stock price trends using Simple Moving Averages (SMA) and Exponential Moving Averages (EMA).
The project includes a Streamlit-powered dashboard for real-time visualization, performance metrics, and trade simulations.

🚀 Features

✅ Fetches live market data using Yahoo Finance (yfinance)
✅ Implements SMA and EMA crossover strategies
✅ Calculates portfolio performance metrics:

Total Return

CAGR (Compound Annual Growth Rate)

Annual Volatility

Sharpe Ratio

Sortino Ratio

Max Drawdown

MAR Ratio

✅ Visualizes equity curves and buy/sell trades using Plotly
✅ Allows dynamic parameter tuning (periods, capital, slippage, etc.)
✅ Interactive Streamlit dashboard for visual analytics and CSV export
✅ Fully extendable for backtesting, position sizing, and risk management

🧠 Tech Stack

Language: Python

Libraries:

pandas, numpy – Data manipulation

yfinance – Market data download

plotly, matplotlib – Visualization

streamlit – Interactive dashboard

Category: Quantitative Finance / Algorithmic Trading

📦 Installation
1️⃣ Clone the repository
git clone https://github.com/yourusername/stock-trend-analysis.git
cd stock-trend-analysis

2️⃣ Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # (Windows)
# or
source venv/bin/activate  # (Mac/Linux)

3️⃣ Install dependencies
pip install -r requirements.txt

💻 Run the Dashboard
streamlit run app_fixed.py


Then open the URL displayed in your terminal, usually:
👉 http://localhost:8501

🧩 Example Tickers

You can analyze U.S. and Indian markets using:

AAPL, MSFT, GOOGL, AMZN, META, TSLA, 
HDFCBANK.NS, RELIANCE.NS, TATAMOTORS.NS, ICICIBANK.NS


Simply enter any of the above in the Ticker input box or select from a dropdown (if implemented).

📈 Example Outputs

The dashboard shows:

SMA & EMA crossover performance

Equity growth chart (vs. Buy & Hold)

Trade logs (with price, shares, and fees)

Performance summary tables

🧪 Sample Metrics Output
Metric	SMA	EMA
Total Return	5.9%	7.0%
CAGR	1.0%	1.2%
Sharpe	0.33	0.35
Max Drawdown	-4.9%	-6.7%
⚙️ Folder Structure
📂 stock-trend-analysis/
│
├── app_fixed.py            # Streamlit dashboard app
├── sma_ema_backtest.py     # Core logic and backtesting engine
├── requirements.txt         # Dependencies
├── README.md                # Project documentation
└── venv/                    # (optional) Virtual environment

🧭 Future Enhancements

🔹 Add position sizing & risk-based allocation

🔹 Include Bollinger Bands or RSI filters

🔹 Backtest portfolio of multiple stocks

🔹 Export results to Excel / PDF report

🔹 Integrate with Alpaca or Zerodha for live paper trading

📚 Learning Outcomes

By completing this project, you’ll strengthen:

Quantitative trading fundamentals

Python data analysis & visualization

Understanding of risk metrics (Sharpe, Sortino, Drawdown)

Real-world backtesting workflows
