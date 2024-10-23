import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as pltdates
from pathlib import Path 
import csv_merger
from csv_columns import WellnessColumns, SleepColumns
from utils import to_datetime

# View the sleep graph of the night between day N and day N+1
# Type: str | Range: 2 <= N <= 31
VIEW_DAY = '31'
# The window around sleep timestamps to check for wellness data.
SLEEP_LOOKAHEAD  = 60 * 60 * 5 # hours
SLEEP_LOOKBEHIND = 60 * 60 * 5 # hours

sleep_path = Path(f"./data/merged/sleep/{VIEW_DAY}.csv")
wellness_path = Path(f"./data/merged/wellness/{VIEW_DAY}.csv")

def get_wellness_df(path: str):
    df = pd.read_csv(path)
    df.rename(columns={
        WellnessColumns.monitoring.steps: 'steps',
        WellnessColumns.monitoring.timestamp_s: 'timestamp',
        WellnessColumns.monitoring.bpm: 'bpm'
    }, inplace=True)

    # Remove spikes of invalid steps data
    # by forcing it to be monotonic ascending    
    df['steps'] = df['steps'].cummax().diff()

    # Remove invalid bpm monitoring data
    df = df[df['bpm'] != 0]
    
    return df

def prepare_data():
    df_sleep = pd.read_csv(sleep_path)
    df_sleep.rename(columns={
        SleepColumns.assessment.timestamp_s: 'timestamp'
    }, inplace=True)

    graph_window = (
        df_sleep['timestamp'].min() - SLEEP_LOOKBEHIND, 
        df_sleep['timestamp'].max() + SLEEP_LOOKAHEAD
    )

    # Wellness data usually goes from the previous day at 21:00 - today at 21:00.
    df = get_wellness_df(wellness_path)

    # Ensure data is within graph window and present
    df: pd.DataFrame = df[df['timestamp'] <= graph_window[1]]
    if df['timestamp'].min() > graph_window[0]:
        prev = int(VIEW_DAY) - 1
        df_previous = get_wellness_df(wellness_path.parent / f"{prev}.csv")
        df_previous = df_previous[df_previous['timestamp'] >= graph_window[0]]
        
        # append df_previous to df
        df = pd.concat([df, df_previous], ignore_index=True).sort_values('timestamp')
    
    df['timestamp'] = df['timestamp'].map(
        lambda t: to_datetime(t)
    )

    return df


if __name__ == '__main__':
    # csv merger has not executed yet
    if not wellness_path.exists():
        csv_merger.generate_data()

    df_wellness = prepare_data()

    # -------------------------------------------------------
    # ---------------------Visualization---------------------

    axis = df_wellness.plot(
        title=f"Steps / Bpm Data 2024 ({VIEW_DAY}.csv)",
        xlabel='Time',
        x='timestamp',
        label=['Heartbeat [bpm]', 'Δ Steps'],
        y=    ['bpm',             'steps'  ]
    )

    # Plot sleep and wakeup times on graph
    df_sleep = pd.read_csv(sleep_path) if sleep_path.exists() else pd.DataFrame() 

    if not df_sleep.empty:
        sleep_timestamps = df_sleep[SleepColumns.assessment.timestamp_s]
        
        sleep_at = to_datetime(sleep_timestamps.min())
        wake_up_at = to_datetime(sleep_timestamps.max())
        plt.axvspan(sleep_at, wake_up_at, color='blue', alpha=0.2, label='Sleep')

    axis.xaxis.set_major_formatter(pltdates.DateFormatter('%m-%d %H:%M'))
    plt.legend()
    plt.show()
