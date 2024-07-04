import yfinance as yf
import datetime as dt
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

def portfolio_i(returns, cov_mat, time):
    a = np.array([np.random.uniform(0,1) for _ in range(cov_mat.shape[0])])
    weights = a/sum(a)*100
    portfolio_return = weights.dot(returns/100)                             #in %
    portfolio_vol = np.sqrt(np.dot(np.dot(weights.T,cov_mat),weights)*np.sqrt(time))    #in % over the whole period (in day)                       
    sharpe = round((portfolio_return-5.33)/portfolio_vol,3)
    return (portfolio_return, portfolio_vol, sharpe, weights)

def generate_portfolios(num, returns, cov_mat, time):
    portfolio_df = pd.DataFrame([portfolio_i(returns, cov_mat, time) for _ in range(num)],
                                columns=['portfolio_return', 'portfolio_vol', 'sharpe', 'weights'])
    return portfolio_df
    
def plot_efficient_frontier(portfolio_df):
    plt.subplots(figsize=(10, 5))
    plt.scatter(x=portfolio_df.portfolio_vol, y=portfolio_df.portfolio_return, c=portfolio_df.sharpe, s=10, cmap='rainbow')
    plt.xlim(0, 8)
    plt.ylim(0, 30)
    plt.ylabel("Expected Portfolio Return")
    plt.xlabel("Portfolio Volatility (Std)")
    plt.title("Efficient Frontier")
    plt.show()

def plot_portfolio_performance(weights, close_values):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))
    
    portfolio_value = (close_values * weights/100).sum(axis=1)
    ax1.plot(portfolio_value.index, portfolio_value, color='purple', alpha=0.7, linewidth=1.0)
    ax1.axhline(y=portfolio_value[0], color='black', linestyle='--', linewidth=.7)
    ax1.set_ylabel('Portfolio Value')

    portfolio_return = portfolio_value.pct_change()
    ax2.hist(portfolio_return, bins=200, color='purple', alpha=0.7)
    ax2.set_ylabel('Frequency')
    ax2.set_xlabel('Portfolio Return')

    plt.tight_layout()
    plt.show()

def plot_compare(weights, pct_changes):
    weighted_pct_change1 = pct_changes.mean(axis=1)
    weighted_pct_change2 = np.dot(pct_changes,weights)/100
    portfolio_value1 = [100]
    portfolio_value2 = [100]
    for i in range(len(weighted_pct_change2)):
        new_value1 = portfolio_value1[-1] * (1 + weighted_pct_change1.iloc[i])
        portfolio_value1.append(new_value1)
        new_value2 = portfolio_value2[-1] * (1 + weighted_pct_change2[i])
        portfolio_value2.append(new_value2)
    plt.plot(pct_changes.index,portfolio_value1[1:], alpha=.7, linewidth=.7, color = 'gray')
    plt.plot(pct_changes.index,portfolio_value2[1:], alpha=.7, linewidth=.7, color = 'purple')
    plt.axhline(y=100, color='black', linestyle='--', linewidth=1.0, label='Initial Portfolio Value')
    plt.ylabel('Portfolio Value')
    plt.tight_layout()
    plt.show()

def var_parametric(portofolioReturns, portfolioStd, alpha=5):
    return norm.ppf(1-alpha/100)*portfolioStd - portofolioReturns

def var_hist(returns, alpha):
    return -np.percentile(returns, q = alpha)/100

if __name__ == "__main__":    
    tickers = ["WAB","LNG","LHX","MDU","TSM","TTE","V","PG","LIN"]
    colours = ["darkred", "lightblue","orangered","green","red","mediumblue","gold","deepskyblue","dodgerblue"]
    ticker_color_mapping = dict(zip(tickers, colours))

    startDate = dt.datetime.strptime("2023-01-01", "%Y-%m-%d")
    endDate = dt.datetime.now()
    nb_days = (endDate - startDate).days
    nb_y = nb_days / 252

    stocks = yf.download(tickers, start=startDate, end=endDate)

    close_values = stocks.loc[:, "Close"]
    close_values = close_values.dropna()

    # close_values.plot(color=[ticker_color_mapping[ticker] for ticker in close_values.columns], title="USD", figsize=(10, 5), linewidth=.7)
    # plt.ylabel("Prices USD")
    # plt.show()

    close_values_nomralized = close_values.div(close_values.iloc[0]).mul(100)
    # close_values_nomralized.plot(color=[ticker_color_mapping[ticker] for ticker in close_values.columns], title = "base 100", figsize = (10, 5), linewidth=.7)
    # plt.axhline(y = 100, color = "black")
    # plt.ylabel("Prices base 100")
    # plt.show()

    pct_change = pd.DataFrame()
    for i, col in enumerate(close_values.columns):
        new_col = f"{col}"
        pct_change[new_col] = close_values[col].pct_change()
    
    returns = pct_change.dropna()
    cumulative_ret = (1 + returns).prod()                           #decimal over whole period
    annualised_ret = (cumulative_ret ** (1 / nb_y) - 1)*100         #in %
    annualised_ret = annualised_ret.rename('Return')
    # print(annualised_ret)

    daily_vol = returns.describe().T.loc[:,["std"]]
    annualised_vol = pd.DataFrame(daily_vol*((nb_days)**0.5)*100)   #in %
    # print(daily_vol,'\n')
    # print(annualised_vol)

    # fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15, 5), sharex=True, sharey=True)
    # axes = axes.flatten()
    # for i, ticker in enumerate(tickers):
    #     col = ticker_color_mapping[ticker]
    #     ax = axes[i]
    #     ax.hist(returns[ticker], bins=100, color=col)
    #     ax.set_xlim(-0.1, 0.1)
    #     ax.set_title(ticker)
    # plt.tight_layout()
    # plt.show()

    # result = pd.concat([annualised_ret, annualised_vol], axis=1)
    # result.plot.scatter(x = "std", y = "Return", color=[ticker_color_mapping[ticker] for ticker in close_values.columns], s = 50, figsize = [10,5])
    # for i in result.index:
    #     plt.annotate(i, xy=(result.loc[i,"std"]+0.005, result.loc[i,"Return"]))
    # plt.xlim(0, 100)
    # plt.ylim(0, 50)
    # plt.ylabel("Asset Return")
    # plt.xlabel("Asset Volatility (Std)")
    # plt.title("CAPM")
    # plt.show()

    portfolios_data = generate_portfolios(1000000, annualised_ret, returns.cov(), nb_days)
    # plot_efficient_frontier(portfolios_data)

    max_portfolio = portfolios_data.loc[portfolios_data['sharpe'].idxmax()]
    print("\nMax Sharpe Ratio:", max_portfolio['sharpe'], "\n\nWeights associated with Max Sharpe Portfolio:\n", max_portfolio['weights'].round(3), "\n")

    # plot_portfolio_performance(max_portfolio['weights'], close_values)
    # plot_compare(max_portfolio['weights'], returns)

    invest = 1000
    alpha = 5
    historical_VaR = var_hist(np.dot(returns,max_portfolio['weights'].T), alpha)
    
    print("Expected Return:      ", max_portfolio['portfolio_return'].round(3), "%\n")
    print(f"Value at Risk {100-alpha}%:   $", (historical_VaR*invest).round(3), '\n')