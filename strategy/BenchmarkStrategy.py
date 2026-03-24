import pandas as pd

def calculate_benchmark_strategy(schedule: pd.DataFrame, date: str, da_sell_threshold: float, limit_price: float = -2) -> dict:
    """
    Calculate the Vattenfall Benchmark Strategy revenue and additional metrics.

    This function calculates the total revenue benchmark based on the strategy of selling:
    - A percentage of forecasted generation in the Day-Ahead market for any price above the limit price.
    - Remaining production in the Intraday market for any price above the limit price.

    Parameters:
    ----------
    schedule : pd.DataFrame
        The schedule DataFrame containing the required columns: 'MCP_DA', 'MCP_ID', 'forecast_gen', and 'actual_prod'.
    date : str
        The date for which the benchmark strategy is calculated (format: 'YYYY-MM-DD').
    da_sell_threshold : float
        The percentage of forecasted generation sold in the Day-Ahead market (e.g., 0.90 for 90%).
    limit_price : float, optional
        The minimum price threshold for selling in both Day-Ahead and Intraday markets (default: -2).

    Returns:
    -------
    dict
        A dictionary containing total benchmark revenue and other relevant metrics.

    Example:
    -------
    metrics = calculate_benchmark_strategy(schedule_df, '2024-10-08', 0.90)
    print(metrics)
    """
    # Filter for Day-Ahead prices above the limit
    filtered_da_benchmark = schedule[schedule['MCP_DA'] > limit_price]
    da_benchmark_bids = da_sell_threshold * filtered_da_benchmark['forecast_gen']
    da_revenue_benchmark = da_benchmark_bids * filtered_da_benchmark['MCP_DA']

    # Filter for Intraday prices above the limit
    filtered_id_benchmark = schedule[schedule['MCP_ID'] > limit_price]
    id_volume_risk = filtered_id_benchmark['actual_prod'] - da_sell_threshold * filtered_id_benchmark['forecast_gen']
    id_revenue_benchmark = id_volume_risk * filtered_id_benchmark['MCP_ID']

    # Calculate total revenue
    total_revenue_benchmark = da_revenue_benchmark.sum() + id_revenue_benchmark.sum()

    return {
        'total_revenue_benchmark'    : total_revenue_benchmark,
        'total_da_benchmark_bids'    : da_benchmark_bids.sum(),
        'total_benchmark_volume_risk': id_volume_risk.sum()
    }
