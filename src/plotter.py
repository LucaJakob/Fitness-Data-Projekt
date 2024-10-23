import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets

from matplotlib.dates import DateFormatter
from csv_columns import SleepColumns, WellnessColumns
from pathlib import Path
from utils import to_datetime

class Figure:
    def __init__(self, day: int):
        self.SLEEP_LOOKAHEAD  = 60 * 60 * 5 # hours
        self.SLEEP_LOOKBEHIND = 60 * 60 * 5 # hours
        
        self.set_day(day)
    
    def set_day(self, day: int):
        self.day = day
        self._sleep_path = Path(f"./data/merged/sleep/{day}.csv")
        self._wellness_path = Path(f"./data/merged/wellness/{day}.csv")

        self.__prepare_data()

    def __prepare_data(self):
        self.df = self.__fetch_wellness(self._wellness_path)
        graph_window = (self.df['timestamp'].min(), self.df['timestamp'].max())

        # Wellness data usually goes from the previous day at 21:00 - today at 21:00.
        # If sleep data is present, focus around that instead
        try:
            self.df_sleep = pd.read_csv(self._sleep_path)
            self.df_sleep.rename(columns={
                SleepColumns.assessment.timestamp_s: 'timestamp'
            }, inplace=True)

            graph_window = (
                self.df_sleep['timestamp'].min() - self.SLEEP_LOOKBEHIND, 
                self.df_sleep['timestamp'].max() + self.SLEEP_LOOKAHEAD
            )
        except Exception:
            self.df_sleep = pd.DataFrame()

        # Wellness data usually goes from the previous day at 21:00 - today at 21:00.
        self.df = self.__fetch_wellness(self._wellness_path)

        # Ensure data is within graph window and present
        self.df: pd.DataFrame = self.df[self.df['timestamp'] <= graph_window[1]]
        if self.df['timestamp'].min() > graph_window[0]:
            prev = self.day - 1
            df_previous = self.__fetch_wellness(self._wellness_path.parent / f"{prev}.csv")
            df_previous = df_previous[df_previous['timestamp'] >= graph_window[0]]
        
            # append df_previous to df
            self.df = pd.concat([self.df, df_previous], ignore_index=True).sort_values('timestamp')
    
        self.df['timestamp'] = self.df['timestamp'].map(to_datetime)


    def __fetch_wellness(self, path: str):
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

    def plot(self, ax: plt.Axes):
        self.df.plot(
        ax=ax,
        title=f"Steps / Bpm Data 2024 ({self.day}.csv)",
        xlabel='Time',
        x='timestamp',
        label=['Heartbeat [bpm]', 'Δ Steps'],
        y=    ['bpm',             'steps'  ]
        )

        if not self.df_sleep.empty:
            sleep_timestamps = self.df_sleep['timestamp']
        
            sleep_at = to_datetime(sleep_timestamps.min())
            wake_up_at = to_datetime(sleep_timestamps.max())
            ax.axvspan(sleep_at, wake_up_at, color='blue', alpha=0.2, label='Sleep')
        ax.legend(loc='upper center')
        ax.xaxis.set_major_formatter(DateFormatter('%m-%d %H:%M'))

class Plotter:
    def __init__(self):
        self.index = 2
        self.index_limit = (2, 31)
        self.current_figure = Figure(self.index)
        fig, ax = plt.subplots()
        self.axes = ax                                                   
        fig.subplots_adjust(top=0.9)
        axis_day = fig.add_axes([0.1, 0.95, 0.8, 0.03])

        self.slider = widgets.Slider(
            axis_day, 'Day', 
            2, 31, valinit=2, valstep=1.0, 
            initcolor='none'
        )
        self.slider.on_changed(self.on_slider)
        self.current_figure.plot(ax)
    
    def on_slider(self, val):
        # Slider is already clamped, so we need no
        # index checks
        self.axes.clear()
        self.index = int(val)
        self.current_figure.set_day(self.index)
        self.current_figure.plot(self.axes)


if __name__ == '__main__':
    # Even though this variable is unused, it prevents
    # Python from garbage collecting our graph.
    # Do not remove this assignment.
    plotter = Plotter()
    plt.show()