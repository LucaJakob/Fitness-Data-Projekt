import pandas             as pd
import matplotlib.pyplot  as plt
import matplotlib.widgets as widgets

from matplotlib.ticker import AutoMinorLocator
from matplotlib.dates  import DateFormatter, MinuteLocator
from csv_columns       import SleepColumns, WellnessColumns
from pathlib           import Path
from utils             import to_datetime, format_title

class Figure:
    def __init__(self, day: int):
        self.LOOKAHEAD  = 60 * 60 * 5 # hours
        self.LOOKBEHIND = 60 * 60 * 5 # hours
        
        self.set_day(day)
    
    def set_day(self, day: int | float):
        self.day = int(day)
        self._sleep_path = Path(f"./data/merged/sleep/{self.day}.csv")
        self._wellness_path = Path(f"./data/merged/wellness/{self.day}.csv")
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
                self.df_sleep['timestamp'].min() - self.LOOKBEHIND, 
                self.df_sleep['timestamp'].max() + self.LOOKAHEAD
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
        title=format_title(self.day),
        xlabel='Time',
        x='timestamp',
        label=['Heartbeat [bpm]', 'Δ Steps'],
        y=    ['bpm',             'steps'  ]
        )

        if not self.df_sleep.empty:
            sleep_timestamps = self.df_sleep['timestamp']
            sleep_at = to_datetime(sleep_timestamps.min())
            wake_up_at = to_datetime(sleep_timestamps.max())
            ax.axvspan(sleep_at, wake_up_at, color='blue', alpha=0.2, label='Sleep', mouseover=True)

        ax.legend(loc='upper center')
        ax.xaxis.set_major_locator(MinuteLocator(byminute=[0, 30]))
        # Keep major tick size for 30 minutes, but remove the text label
        for label in ax.xaxis.get_ticklabels():
            if ':30' in label.get_text():
                label.set_visible(False)
        ax.xaxis.set_major_formatter(DateFormatter('%I %p'))

        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.xaxis.set_tick_params(labelrotation=55)


class Plotter:
    def __init__(self):
        self.index_limit = (2, 31)
        self.graph = Figure(2)
        fig, ax = plt.subplots()
        self.axes = ax
        self.fig = fig                                                   
        fig.subplots_adjust(top=0.9)
        axis_day = fig.add_axes([0.1, 0.95, 0.8, 0.03])

        self.slider = widgets.Slider(
            axis_day, 'Day', 
            2, 31, valinit=2, valstep=1.0, 
            initcolor='none'
        )
        self.slider.on_changed(self.on_slider_change)
        fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.graph.plot(ax)
    
    def on_slider_change(self, val):
        self.axes.clear()
        self.graph.set_day(val)
        self.graph.plot(self.axes)
        self.fig.canvas.draw_idle()
    
    def on_key_press(self, event):
        # Slider.set_val() triggers the Slider.on_change event.
        if event.key == 'right' and self.slider.val != self.index_limit[1]:
            self.slider.set_val(self.slider.val + 1)
        elif event.key == 'left' and self.slider.val != self.index_limit[0]:
            self.slider.set_val(self.slider.val - 1)



if __name__ == '__main__':
    # Even though this variable is unused, it prevents
    # Python from garbage collecting our graph.
    # Do not remove this assignment.
    plotter = Plotter()
    plt.show()