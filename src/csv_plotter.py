import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as pltdates
import matplotlib.axes as pltaxes
from pathlib import Path 
from csv_merger import file_merger_days
from csv_columns import MergedColumns

VIEW_CSV = '25.csv'
TWENTY_YEARS_S = 631152000

x_axis_format = pltdates.DateFormatter('%Y-%m-%d %H:%M')


# def df_replace(col_label, condition, replace_with, data_frame):
#     data_frame.loc[
#          condition(data_frame[col_label]), 
#          col_label
#      ] = replace_with


if __name__ == '__main__':

    merged_dir = Path('./data/merged')

    # csv merger has not executed yet
    if not merged_dir.is_dir():
        file_merger_days()
    df = pd.read_csv(merged_dir / VIEW_CSV)

    bpm_col = MergedColumns.monitoring.bpm
    steps_col = MergedColumns.monitoring.steps
    timestamp_col = MergedColumns.monitoring.timestamp_s

    # Remove spikes of invalid steps data
    # by forcing it to be monotonic ascending    
    df['diff_steps'] = df[steps_col].cummax().diff()

    df[timestamp_col] = df[timestamp_col].map(
        lambda t: pd.to_datetime(t + TWENTY_YEARS_S, unit='s')
    )
    axes: pltaxes.Axes = df.plot(
        x=timestamp_col, 
        xlabel="Time",
        y=['diff_steps', bpm_col]
    )

    axes.xaxis.set_major_formatter(x_axis_format)    
    plt.show()
