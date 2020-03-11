import pandas as pd
import numpy as np


def build_portfolio(df_price: pd.DataFrame, df_cap: pd.DataFrame) -> pd.Series:
    """

    :param df_price: pandas dataframe with close price data
    :param df_cap: pandas dataframe with market cap data
    :return: pandas series with portfolio performace
    """
    # Transform Date column to datetime index
    df_price['Date'] = pd.to_datetime(df_price['Date'])
    df_price = df_price.set_index('Date')
    df_cap['Date'] = pd.to_datetime(df_cap['Date'])
    df_cap = df_cap.set_index('Date')
    # Check if there enought data
    assert(df_price.index[0] > df_cap.index[0]
           ), 'First date for market cap must be earlier then for close'
    # Prepare dataframe with 1 day returns (for PnL calculation)
    df_returns = df_price.divide(df_price.shift(1)) - 1
    df_returns = df_returns.replace([np.inf, -np.inf, None], np.nan)
    # Prepare list of 20 days returns with 1-day delay
    df_returns20 = df_price.divide(df_price.shift(20)).shift(1) - 1
    df_returns20 = df_returns.replace([np.inf, -np.inf, None], np.nan)
    # Create empty dataframe for portfolio weights
    df_weights = pd.DataFrame().reindex_like(df_price)
    # Initial book = 1
    booksize = 1
    # Start portfolio simulation from first day in Price dataframe
    day_curr = pd.Timestamp('1900-01-01T12')
    for day in df_price.index:
        # If today is the fisrt business day in month  or it is initial day
        # we rebalance our portfolio
        if day.year > day_curr.year or day.month > day_curr.month:
            day_curr = day
            # Get TOP10 stocks by market cap using last available data
            # If cap data was udated today - use previous value (to avoid bias)
            df_cap_curr = df_cap.loc[df_cap.index < day_curr].iloc[-1]
            top_10_curr = df_cap_curr.nlargest(10).index
            # Get current 20 days returns
            returns20_curr = df_returns20.loc[day_curr, top_10_curr]
            # If at least one 20 days returns is not NaN, we use it as weight
            # Otherwise we form equal-weighted portfolio
            if returns20_curr.notna().any():
                # Formula for weights (assuming that short sales are allowed):
                #  weight[i] = returns20[i] / sum_i(abs(returns20))
                df_weights.loc[day_curr, top_10_curr] = returns20_curr / \
                    abs(returns20_curr).sum(skipna=True) * booksize
                df_weights.loc[day_curr].fillna(0, inplace=True)
            else:
                df_weights.loc[day_curr, top_10_curr] = 1 / \
                    len(top_10_curr) * booksize
                df_weights.loc[day_curr].fillna(0, inplace=True)
    # Forwardfill weights in portfolio
    df_weights = df_weights.fillna(method='ffill')
    # Calculate PnL  as yesterday's weights * today's returns
    result_pnl = (df_returns * df_weights.shift(1)).sum(skipna=True, axis=1)
    result = result_pnl.cumsum() + booksize
    return result
