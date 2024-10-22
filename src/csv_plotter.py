import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as pltdates
import matplotlib.axes as pltaxes
from pathlib import Path 
import csv_merger
from csv_columns import WellnessColumns, SleepColumns
from utils import to_datetime

VIEW_DAY = '25'


if __name__ == '__main__':

    merged_dir = Path('./data/merged/wellness')

    # csv merger has not executed yet
    if not merged_dir.is_dir():
        csv_merger.generate_data()

    df_wellness = pd.read_csv(merged_dir / f"{VIEW_DAY}.csv")

    bpm_col = WellnessColumns.monitoring.bpm
    steps_col = WellnessColumns.monitoring.steps
    timestamp_col = WellnessColumns.monitoring.timestamp_s

    # Remove spikes of invalid steps data
    # by forcing it to be monotonic ascending    
    df_wellness[steps_col] = df_wellness[steps_col].cummax().diff()

    df_wellness[timestamp_col] = df_wellness[timestamp_col].map(
        lambda t: to_datetime(t)
    )

    # -------------------------------------------------------
    # ---------------------Visualization---------------------


    axis = df_wellness.plot(
        title=f"Steps / Bpm Data 2024 ({VIEW_DAY}.csv)",
        xlabel='Time',
        x=timestamp_col,
        y=    [ bpm_col,           steps_col  ], 
        label=['Heartbeat [bpm]', 'Δ Steps'   ]
    )

    # Plot sleep and wakeup times on graph
    df_sleep = pd.read_csv(f"./data/merged/sleep/{VIEW_DAY}.csv") if Path(f"./data/merged/sleep/{VIEW_DAY}.csv").exists() else pd.DataFrame() 

    if not df_sleep.empty:
        sleep_timestamps = df_sleep[SleepColumns.assessment.timestamp_s]
        
        sleep_at = to_datetime(sleep_timestamps.min())
        wake_up_at = to_datetime(sleep_timestamps.max())
        plt.axvspan(sleep_at, wake_up_at, color='blue', alpha=0.2, label='Sleep')

    axis.xaxis.set_major_formatter(pltdates.DateFormatter('%m-%d %H:%M'))
    plt.legend()
    plt.show()
