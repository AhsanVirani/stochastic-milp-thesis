import os
import pandas as pd
from datetime import datetime, timedelta

def load_csvs_with_substring(directory, substring):
    """
    Load all CSV files from the specified directory whose filenames contain the given substring.
    Remove the first column from each CSV and store the resulting DataFrames in a list.
    Sort both the DataFrames and dates by the dates.

    Parameters:
    - directory (str): The path to the directory containing the CSV files.
    - substring (str): The substring to match in the filenames.

    Returns:
    - List[pd.DataFrame]: A sorted list of pandas DataFrames with the first column removed.
    - List[str]: A sorted list of dates extracted from the filenames.
    """
    dataframes = []
    dates = []
    for filename in os.listdir(directory):
        if filename.endswith('.csv') and substring in filename:
            filepath = os.path.join(directory, filename)
            df = pd.read_csv(filepath)
            if not df.empty:
                df = df.iloc[:, 1:]  # Remove the first column
                dataframes.append(df)
                dates.append(filepath.split('_')[-1][:-4])  # Extract the date part from the filename
    
    # Combine DataFrames and dates into a list of tuples and sort by dates
    sorted_pairs = sorted(zip(dates, dataframes), key=lambda x: x[0])
    
    # Unpack the sorted tuples back into separate lists
    sorted_dates, sorted_dataframes = zip(*sorted_pairs)
    
    return list(sorted_dataframes), list(sorted_dates)


def generate_dates(start_date, end_date):
    """
    Generate a list of dates in 'YYYY-MM-DD' format between the start_date and end_date (inclusive).

    :param start_date: The start date as a string in 'YYYY-MM-DD' format.
    :param end_date: The end date as a string in 'YYYY-MM-DD' format.
    :return: A list of dates in 'YYYY-MM-DD' format.
    """
    # Parse the start and end dates
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    # Generate the dates
    date_list = [(start + timedelta(days=i)).strftime("%Y-%m-%d") 
                 for i in range((end - start).days + 1)]
    return date_list
