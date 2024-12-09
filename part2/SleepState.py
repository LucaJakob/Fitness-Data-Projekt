import pandas as pd
from datetime import date
from utils import read_all_csv, to_datetime

class SleepState:
  __loaded = False
  __map: dict[int, int] = {}
  __limits: dict[int, (int, int)] = {}

  @staticmethod
  def __load():
    """
Load sleep data to this class's mapper.
    """
    for i in range(1, 32):
      all_df = read_all_csv(f"data/2024-08/CSV/CSV_DATA/{i}/SLEEP/*.csv")
      if len(all_df) == 0:
        continue
      SleepState.__parse_df(all_df, i)

    for i in range(1, 32):
      all_df = read_all_csv(f"data/2024-09/CSV/CSV_DATA/{i}/SLEEP/*.csv")
      if len(all_df) == 0:
        continue
      SleepState.__parse_df(all_df, i)

    SleepState.__sort_map()
    SleepState.__loaded = True
  
  @staticmethod
  def __parse_df(df_list: list[pd.DataFrame], day: int):
    """
Parse the provided list of dataframes and load them into the mapper.
    """
    col_level = 'sleep_level.sleep_level'
    col_time = 'sleep_level.timestamp[s]'

    for df in df_list:
      # We only care about rows where both values are present
      df = df[(df[col_level].notna()) & (df[col_time].notna())]

      for _, row in df.iterrows():
          if SleepState.__map.get(row[col_time]) is not None:
            val = SleepState.__map[row[col_time]]
            if row[col_level] == val:
              continue
            print("Duplicate timestamp detected")

          val_limits = (pd.Timestamp.today().timestamp(), pd.Timestamp(2000, 1, 1).timestamp())
          start, stop = SleepState.__limits.get(day, val_limits)
          SleepState.__limits[day] = (min(start, row[col_time])), max(stop, row[col_time])

          SleepState.__map[row[col_time]] = row[col_level]


  @staticmethod
  def __sort_map():
    """
Sort the internal map by key in ascending order.
    """
    SleepState.__map = { 
      k: v for k, v in sorted(
        SleepState.__map.items(), 
        key = lambda p: p[0])
    }

  @staticmethod
  def is_during_sleep(timestamp: int) -> bool:
    """
    Determines whether or not a provided timestamp is within a sleep window.
    """
    # technically doable in a one-liner, but it's hard to read
    for start, stop in SleepState.__limits.values():
      if timestamp < start or timestamp > stop:
        continue
      return True
    return False

  @staticmethod
  def get(timestamp: int) -> int:
    """
    Get the sleep level associated with the provided timestamp.
    An integer between 1 and 4 representing the sleep states is returned.

    1: Awake
    2: Light Sleep
    3: Deep Sleep
    4: REM
    """
    if not SleepState.__loaded:
      SleepState.__load()
    
    if not SleepState.is_during_sleep(timestamp):
      return 1
    
    for k in SleepState.__map.keys():
      if timestamp > k:
        continue
      return SleepState.__map[k]
    return 1 # should be unreachable
  
