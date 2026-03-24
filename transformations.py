import pandas as pd
import numpy as np

# Variance remains a problem and we want to stabilize that. We can't use logs because of negative prices and RDL.
# We apply hyperbolic area sine transformation
# Source: Day-Ahead Electricty Price Forecasting during Periods of Increased Volatility. Available at: https://www.diva-portal.org/smash/get/diva2:1739566/FULLTEXT02

def transform_prices(prices, training_window=None):
    """
    Transform prices using (median, MAD) normalization with arsinh transformation.
    
    Parameters:
    - prices: pandas Series, the original time series of prices (indexed by datetime).
    - training_window: int or None, the number of observations (hours) to calculate the median and MAD.
      If None, a rolling window approach will be used.
    - lag: int, the lag to shift the series to avoid lookahead bias.
    
    Returns:
    - transformed_prices: pandas Series, the arsinh transformed prices.
    - median: float, median used for normalization.
    - mad: float, MAD used for normalization.
    """
    # Step 1: Calculate Median and MAD based on training window or rolling
    if training_window:
        # Use full training window for normalization
        median = prices.iloc[:training_window].median()
        mad = np.median(np.abs(prices.iloc[:training_window] - median)) * 1.4826
    else:
        # Use expanding median and MAD if training_window is not specified
        median = prices.expanding().median()
        mad = prices.expanding().apply(lambda x: np.median(np.abs(x - np.median(x)))) * 1.4826
    
    # Step 2: Normalize and Apply arsinh Transformation
    normalized_prices = (prices - median) / mad
    transformed_prices = np.arcsinh(normalized_prices)
    
    return transformed_prices, median, mad

def inverse_transform(transformed_prices, median, mad):
    """
    Inverse the arsinh transformation to obtain raw forecasted prices.
    
    Parameters:
    - transformed_prices: pandas Series, the forecasted values in arsinh-transformed space.
    - median: float, median used for normalization.
    - mad: float, MAD used for normalization.
    
    Returns:
    - raw_prices: pandas Series, the raw forecasted price values.
    """
    # Step 1: Inverse the arsinh transformation
    normalized_prices = np.sinh(transformed_prices)
    
    # Step 2: Reverse the normalization
    raw_prices = (normalized_prices * mad) + median
    
    return raw_prices

# # #training window len 
# training_window = len(prices.iloc[1:-1006,:])
# transformed_da_prices, median_da_prices, mad_da_prices = transform_prices(prices['DA_price'], training_window)
# transformed_rl_f, median_rl_f, mad_rl_f = transform_prices(rl_f, training_window)