import pandas as pd
import part2.utils as utils
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import glob
import math

class SleepState:
    Level = {
        "Awake": 1,
        "Light": 2,
        "Deep": 3,
        "REM": 4
    }

    __timestamps = {}
    __map = {}
    __limits = {}

    def __init__(self, path: str, day: int):
        all_files = glob.glob(rf'{path}/{day}/SLEEP/*.csv',recursive=True)
        all_df = [pd.read_csv(f, sep=',') for f in all_files]
        
        SleepState.__timestamps[day] = []
        for df in all_df:
            for _, row in df.iterrows():
                SleepState.__append(row, day)
                 
    @staticmethod
    def __append(row: pd.Series, day: int):
        ts = row['sleep_level.timestamp[s]']
        level = row['sleep_level.sleep_level']
        if pd.isna(ts) or pd.isna(level):
            return

        SleepState.__map[ts] = level
        SleepState.__timestamps[day].append(ts)
        SleepState.__timestamps[day].sort()

        start, stop = SleepState.__limits.get(day, (datetime(2030, 1, 1).timestamp(), 0))

        limits = (min(start, ts), max(stop, ts))
        SleepState.__limits[day] = limits

    @staticmethod
    def is_within_limits(timestamp):
        for i in SleepState.__limits.values():
            if timestamp >= i[0] and timestamp <= i[1]:
                return True
        return False

    @staticmethod
    def get(day: int, time: int):
        try:
            ts = next(ts for ts in SleepState.__timestamps[day] if ts >= time)
            level = SleepState.__map.get(ts)
            return int(level)
        except StopIteration:
            return None


table = pd.read_csv('data/2024-08/CSV/CSV_DATA/2/SLEEP/269633116559_SLEEP_DATA_data.csv')

hrv_table = pd.read_csv('data/2024-08/CSV/CSV_DATA/2/HRV/269633110015_HRV_STATUS_data.csv')

df = pd.DataFrame()
df['timestamp'] = table['sleep_level.timestamp[s]'].map(utils.to_datetime).map(lambda t: t + pd.offsets.Hour(2))
df['sleep_level'] = table['sleep_level.sleep_level']
df = df[(df['timestamp'].notna()) & (df['sleep_level'].notna())]

fig, ax = plt.subplots()

def append_data(df: pd.DataFrame, row: pd.Series, timestamp_key: str, columns: list):
    ts = row[timestamp_key]
    if pd.isna(ts):
        return
    
    if not SleepState.is_within_limits(ts):
        return
    
    if ts in df['timestamp'].values:
        pass
    else:
        pass


def is_duplicate():
    pass


if __name__ == '__main__':
    for i in range(1, 32):
        SleepState("data/2024-08/CSV/CSV_DATA", i)
    
    df = pd.DataFrame()
    for i in range(1, 32):
        all_files = glob.glob(rf'data/2024-08/CSV/CSV_DATA/{i}/HRV/*.csv',recursive=True)
        all_hrv = [pd.read_csv(f, sep=',') for f in all_files]

        for table in all_hrv:
            for _, row in table.iterrows():
                append_data(df, row, 'hrv_value.timestamp')

    print(SleepState.getDict().keys())

# df.plot(ax=ax, x='timestamp',y='sleep_level', kind='scatter' )
# wake_time = datetime(2024,8,3,3,10,0)
# sleep_time = utils.to_datetime(table['event.timestamp[s]'].unique()[1]) + pd.offsets.Hour(2)
# plt.axvline(x=wake_time)
# plt.axvline(x=sleep_time)

# plt.step(x=df['timestamp'], y=df['sleep_level'])
# plt.show()

# Point marks end time of last interval
# First assumption is Light Sleep

# enum:
# Awake: 1
# Light sleep: 2
# Deep sleep: 3
# REM: 4



# Best approach:
# -> timestamp >= Sleep time, timestamp <= wake time
# -> All timestamps -> fill with next big value of timestamp sleep

# 3am Level 2
# 2:30am Some data -> level 2
# 3:10am Some data -> Level ???

# AI -> Training data sind alle Daten
# Input -> Sleep time, Wake time
# Output -> 1, 2, 3, 4 for every 60s


# Wellness:
# stress_level.stress_level_value
# respiration_rate.respiration_rate[breaths/min]
# monitoring.heart_rate[bpm]
# monitoring.steps[steps]

# Based on: respiration_rate.timestamp