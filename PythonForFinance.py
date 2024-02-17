import yfinance as yf
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
import numpy as np
import math

def portfolio_i(returns, cov_mat):
    a = np.array([np.random.uniform(0,1) for _ in range(cov_mat.shape[0])])
    weights = a/sum(a)
    portfolio_return = weights.dot(returns)
    portfolio_cov = np.dot(np.dot(weights.T,cov_mat),weights)*252*5*100
    portfolio_vol = math.sqrt(portfolio_cov)
    sharpe = round((portfolio_return-2.463)/portfolio_vol,3)
    return (portfolio_return, portfolio_cov, portfolio_vol, sharpe, weights)

def generate_portfolios(num, returns, cov_mat):
    portfolio_df = pd.DataFrame([portfolio_i(returns, cov_mat) for _ in range(num)],
                                columns=['portfolio_return', 'portfolio_cov', 'portfolio_vol', 'sharpe', 'weights'])
    return portfolio_df

def plot_efficient_frontier(portfolio_df):
    plt.subplots(figsize=(10, 5))
    plt.scatter(x=portfolio_df.portfolio_vol, y=portfolio_df.portfolio_return, c=portfolio_df.sharpe, s=15, cmap='rainbow')
    plt.xlim(0, 8)
    plt.ylim(0, 30)
    plt.ylabel("Expected Portfolio Return")
    plt.xlabel("Portfolio Volatility (Std)")
    plt.title("Efficient Frontier")
    plt.show()

def find_max_sharpe_ratio(portfolio_df):
    return portfolio_df.loc[portfolio_df['sharpe'].idxmax(), ['sharpe', 'weights']] * (1, 100)

def calculate_portfolio_value(weights, close_values_nomralized):
    return (close_values_nomralized * weights/100).sum(axis=1)

def plot_portfolio_performance(weights, close_values_nomralized):
    portfolio_value = calculate_portfolio_value(weights, close_values_nomralized)
    portfolio_return = portfolio_value.pct_change()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))

    ax1.plot(portfolio_value.index, portfolio_value, color='purple', alpha=0.7, linewidth=1.0)
    ax1.axhline(y=100, color='black', linestyle='--', linewidth=.7)

    ax1.set_ylabel('Portfolio Value')

    ax2.hist(portfolio_return, bins=200, color='purple', alpha=0.7)
    ax2.set_ylabel('Frequency')
    ax2.set_xlabel('Portfolio Return')

    plt.tight_layout()
    plt.show()

def plot_compare(weights_list, close_values_normalized):
    line_colors = ['gray', 'purple']
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, weights in enumerate(weights_list):
        portfolio_value = calculate_portfolio_value(weights, close_values_normalized)
        portfolio_return = portfolio_value.pct_change()
        color = line_colors[i % len(line_colors)]
        plt.plot(portfolio_value.index, portfolio_value, alpha=.7, linewidth=.7, color=color)

    plt.axhline(y=100, color='black', linestyle='--', linewidth=1.0, label='Initial Portfolio Value')
    ax.set_ylabel('Portfolio Value')
    plt.tight_layout()
    plt.show()

tickers = ["WAB","LNG","LHX","MDU","TSM","TTE","V","NVDA","PG","LIN"]
colours = ["darkred", "lightblue","orangered","green","red","mediumblue","gold","lime","deepskyblue","dodgerblue"]
stocks = yf.download(tickers, start = "2018-01-01", end = "2023-01-01")   #YYYY-MM-DD
ticker_color_mapping = dict(zip(tickers, colours))
close_values = stocks.loc[:, "Close"]
close_values = close_values.dropna()

close_values.plot(color=[ticker_color_mapping[ticker] for ticker in close_values.columns], title="USD", figsize=(10, 5), linewidth=.7)
plt.ylabel("Prices USD")
plt.show()

close_values_nomralized = close_values.div(close_values.iloc[0]).mul(100)
close_values_nomralized.plot(color=[ticker_color_mapping[ticker] for ticker in close_values.columns], title = "base 100", figsize = (10, 5), linewidth=.7)
plt.axhline(y = 100, color = "black")
plt.ylabel("Prices base 100")
plt.show()

pct_change = pd.DataFrame()
for i, col in enumerate(close_values.columns):
    new_col = f"{col}"
    pct_change[new_col] = close_values[col].pct_change()
returns = np.log(1 + pct_change)
returns = returns.dropna()
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15, 5), sharex=True, sharey=True)
axes = axes.flatten()
for i, ticker in enumerate(tickers):
    col = ticker_color_mapping[ticker]
    ax = axes[i]
    ax.hist(returns[ticker], bins=200, color=col)
    ax.set_xlim(-0.1, 0.1)
    ax.set_title(ticker)
plt.tight_layout()
plt.show()

daily_vol = returns.describe().T.loc[:,["std"]]
annualised_vol = pd.DataFrame(daily_vol*((252*5)**0.5)*100)
og_value = close_values.iloc[0, :]
last_px = close_values.iloc[-1, :]
cumulative_returns = pd.DataFrame((last_px - og_value) / og_value)
annualised_ret = pd.DataFrame((((1+cumulative_returns)**(365/len(returns)))-1)*100)
annualised_ret = annualised_ret.rename(columns={0: 'Return'})

result = pd.concat([annualised_ret, annualised_vol], axis=1)
result.plot.scatter(x = "std", y = "Return", color=[ticker_color_mapping[ticker] for ticker in close_values.columns], s = 50, figsize = [10,5])
for i in result.index:
    plt.annotate(i, xy=(result.loc[i,"std"]+0.005, result.loc[i,"Return"]))
plt.xlim(0, 150)
plt.ylim(0, 50)
plt.ylabel("Asset Return")
plt.xlabel("Asset Volatility (Std)")
plt.title("CAPM")

portfolios_data = generate_portfolios(100000, result["Return"], np.array(returns.cov()))
plot_efficient_frontier(portfolios_data)

max_sharpe_ratio, max_sharpe_weights = find_max_sharpe_ratio(portfolios_data)
max_sharpe_portfolio_value = calculate_portfolio_value(max_sharpe_weights, close_values_nomralized)

plot_portfolio_performance(max_sharpe_weights, close_values_nomralized)
print("Max Sharpe Ratio:", max_sharpe_ratio, "\nWeights associated with Max Sharpe Portfolio:\n", max_sharpe_weights)

benchmark = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
MoPoTh = max_sharpe_weights

plot_compare([benchmark, MoPoTh], close_values_nomralized)