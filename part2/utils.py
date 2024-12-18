import pandas as pd
import glob
import datetime
from pathlib import Path

def read_all_csv(regex_like_path: str, index_col: str | None = None) -> list[pd.DataFrame]:
    """
    Read all CSV based on the provided path expression.
    Use * to mark any string.
    Example: 'src/*.csv'
    """
    all_files = glob.glob(regex_like_path, recursive=True)
    
    if index_col is None:
        return [pd.read_csv(f) for f in all_files]
    
    dfs = []

    for f in all_files:
        try:
            dfs.append(pd.read_csv(f, index_col=index_col))
        except ValueError: 
            continue
    return dfs

def to_datetime(garmin_timestamp: int) -> pd.Timestamp:
    """
Converts a Garmin timestamp into a datetime object. Garmin timestamps differ from Unix
because they start at Jan 1, 1990 instead of Unix's 1970.

Parameters
----------

garmin_timestamp : int
    The timestamp that originated from a garmin device.

    """

    d = pd.to_datetime(garmin_timestamp, unit='s')

    # Garmin timestamps begin at Jan 1, 1990. To ensure that leap years do not
    # Break or change the units below years, we use the code below.
    # See: 
    #   https://stackoverflow.com/questions/32799428/adding-years-in-python
    #   https://stackoverflow.com/questions/48796729/how-to-add-a-year-to-a-column-of-dates-in-pandas
    return d + pd.offsets.DateOffset(years=20)

def format_title(day: int) -> str:
    """
Format the visual graph's title based on the provided day.
Assumes that the CSV file is generated and named with the same integer.
    """
    title = "Aug 31 - Sep 01" if day == 31 else f"Aug {day} - Aug {day + 1}"
    title += f" 2024 ({day}.csv)"
    return title

def read_wellness_csv(
        path: Path | str
    ) -> pd.DataFrame:
    """
Reads the wellness CSV file at the provided path,
and formats the data to be more digestible.
This adjusts the column types such as datetime ones.

Parameters
----------

path : pathlib.Path | str
    The path to use to read the CSV. Ensure the provided file is a wellness-like file.

    """
    df = pd.read_csv(path)
    df['timestamp'] = df['timestamp'].map(to_datetime)
    
    return df

def read_sleep_csv() -> pd.DataFrame:
    sleep_df = pd.read_csv('data/2024-08/CSV/SLEEP/sleep_data.csv')

    def apply_func(row):
        if not isinstance(row['wake_time'], str):
            return pd.NA
        date_str = f"{row['date']} {row['wake_time']}"
        # Sleep data timestamps are 3 hours ahead of steps data, MAYBE because of 12am-12am measurements
        # instead of 9pm-9pm step measurements
        # This data has been double checked using the garmin's interpreted data
        return datetime.datetime.strptime(date_str, "%Y-%m-%d %H:%M %p") - pd.offsets.Hour(3)

    sleep_df['wake_time'] = sleep_df.apply(apply_func, axis=1)

    sleep_df['duration'] = sleep_df.apply(
        lambda row: pd.to_timedelta(row['duration']),
        axis=1
    )

    sleep_df['bedtime'] = sleep_df.apply(
        lambda row: row['wake_time'] - row['duration'],
        axis=1
    )
    return sleep_df