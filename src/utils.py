import pandas as pd

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