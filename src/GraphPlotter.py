import datetime
import pandas as pd
from pathlib import Path
from utils   import to_datetime, format_title, read_sleep_csv

import matplotlib.pyplot   as plt
import matplotlib.widgets  as widgets
from   matplotlib.ticker   import AutoMinorLocator
from   matplotlib.dates    import DateFormatter, MinuteLocator

class GraphFigure:
    def __init__(self, day: int):
        self.LOOKAHEAD  = pd.offsets.Hour(5)
        self.LOOKBEHIND = pd.offsets.Hour(5)
        
        self.df_sleep = read_sleep_csv()

        self.set_day(day)
    
    def set_day(self, day: int | float):
        self.day = int(day)
        self._wellness_path = Path(f"./data/merged/wellness/{self.day}.csv")
        self.__prepare_data()

    def __prepare_data(self):
        self.df = self.__fetch_wellness(self._wellness_path)

        self.graph_window = (self.df['timestamp'].min(), self.df['timestamp'].max())

        # Wellness data usually goes from the previous day at 21:00 - today at 21:00.
        # Find the first wake up time that is between those two dates
        row_index = self.df_sleep.index[
            (self.df_sleep['wake_time'] < self.graph_window[1]) & 
            (self.df_sleep['wake_time'] > self.graph_window[0])
        ]

        if row_index.empty:
            self.sleep_time = None
            self.wake_time = None
        else:
            self.sleep_time = pd.Timestamp(self.df_sleep.loc[row_index[0]]['bedtime'])
            self.wake_time = pd.Timestamp(self.df_sleep.loc[row_index[0]]['wake_time'])
            upper = self.wake_time + self.LOOKAHEAD
            lower = self.sleep_time  - self.LOOKBEHIND
            self.graph_window = (lower, upper)

        # Wellness data usually goes from the previous day at 21:00 - today at 21:00.
        self.df = self.__fetch_wellness(self._wellness_path)

        # Ensure data is within graph window and present
        self.df = self.df[self.df['timestamp'] <= self.graph_window[1]]
        if self.df['timestamp'].min() > self.graph_window[0]:
            prev = self.day - 1
            df_previous = self.__fetch_wellness(self._wellness_path.parent / f"{prev}.csv")
            df_previous = df_previous[df_previous['timestamp'] >= self.graph_window[0]]
        
            # append df_previous to df
            self.df = pd.concat([self.df, df_previous], ignore_index=True).sort_values('timestamp')
    

    def __fetch_wellness(self, path: str):
        df = pd.read_csv(path)
        df['timestamp'] = df['timestamp'].map(to_datetime)
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

        # Mark sleep span
        if self.sleep_time is not None and self.wake_time is not None:
            ax.axvspan(
                self.sleep_time, 
                self.wake_time,
                color='blue', alpha=0.2, label='Sleep'
            )

        # Add dotted line at midnight
        date = self.df['timestamp'].max()
        midnight_val = datetime.datetime(date.year, date.month, date.day)
        ax.axvline(midnight_val, color='black', linestyle='--')

        ax.legend(loc='upper center')

        # Label every hour
        # Big tick every 30 minutes
        # Small tick every 15 minutes
        ax.xaxis.set_major_locator(MinuteLocator(byminute=[0, 30]))
        for label in ax.xaxis.get_ticklabels():
            if ':30' in label.get_text():
                label.set_visible(False)
        ax.xaxis.set_major_formatter(DateFormatter('%I %p'))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.xaxis.set_tick_params(labelrotation=55)



class Plotter:
    def __init__(self):
        self.index_limit = (2, 31)
        self.graph = GraphFigure(2)
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