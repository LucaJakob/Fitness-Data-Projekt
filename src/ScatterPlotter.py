import pandas as pd
from pathlib import Path
import datetime
from part2.utils import read_wellness_csv
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from matplotlib.ticker import AutoMinorLocator
from matplotlib.dates  import DateFormatter, MinuteLocator
import csv_merger

class ScatterFigure:
    def __init__(self):
        """
Creates a line graph Figure at the initial day.
        """
        cols = ['day', 'sleep_length', 'wake_time']
        cols.extend([f"mean_{i}" for i in range(16)])

        self.df = pd.DataFrame(columns=cols)
        self.sleep_df = pd.read_csv('data/2024-08/CSV/SLEEP/sleep_data.csv')
        self.__format_sleep_data()

        self.df['sleep_length'] = self.sleep_df['duration']
        self.df['day'] = self.sleep_df.apply(lambda row: row['wake_time'].day, axis=1)
        self.df['wake_time'] = self.sleep_df['wake_time']
        
        self.__calculate_steps_mean()
        self.current_mean = None
        
        # timedelta is not supported by matplotlib
        self.df['sleep_length'] = self.df['sleep_length'].map(lambda t: pd.to_datetime(t.total_seconds(), unit="s"))

        

    def __format_sleep_data(self):
        self.sleep_df['wake_time'] = self.__col_to_datetime('wake_time')
        # Wake time is required to calculate the steps' mean, so drop
        # useless rows
        self.sleep_df = self.sleep_df.dropna(subset=['wake_time'])
        self.sleep_df['duration'] = self.sleep_df.apply(
            lambda row: pd.to_timedelta(row['duration']),
            axis=1
        )
    
    def __col_to_datetime(self, col):
        def apply_func(row):
            if not isinstance(row[col], str):
                return pd.NA
            date_str = f"{row['date']} {row[col]}"
            return datetime.datetime.strptime(date_str, "%Y-%m-%d %H:%M %p")

        return self.sleep_df.apply(apply_func, axis=1)

    def __calculate_steps_mean(self):
        csv_dir = Path('data/merged/wellness')
        for date_csv in csv_dir.iterdir():
            if date_csv.suffix != '.csv':
                continue
            # csv dates go from {d} 9pm - {d + 1} 9pm
            date = int(date_csv.stem) + 1
            row_index = self.df.index[self.df['day'] == date]

            # Sleep data at this particular day had no data
            if row_index.empty:
                continue
            elif len(row_index) > 1:
                raise ValueError('DataFrame contains duplicate rows of days.')
            
            wellness_df = read_wellness_csv(date_csv)
            lower = self.df.loc[row_index[0]]['wake_time']

            for i in range(16):
                # max: 8 hours
                window = pd.offsets.Minute(i * 30 + 30)
                upper = lower + window
                
                steps_data = wellness_df[wellness_df['timestamp'] >= lower]
                steps_data = steps_data[steps_data['timestamp'] <= upper]

                # Data has been lost while monitoring
                if steps_data.empty:
                    continue
                
                self.df.at[row_index[0], f"mean_{i}"] = steps_data['steps'].mean()
     
    def __get_axis_limits(self) -> tuple[tuple[str, str], tuple[str, str]]:
        x_offset = pd.offsets.Minute(30)
        xmin = self.df['sleep_length'].min() - x_offset
        xmax = self.df['sleep_length'].max() + x_offset

        ymin = 0
        ymax = self.df['mean_0'].max()

        for i in range(1, 16):
            ymax = max(self.df[f"mean_{i}"].max(), ymax)
        
        return ((xmin, xmax), (ymin, ymax))
    
    def plot(self):
        fig, ax = plt.subplots()
        self.axes = ax
        self.fig = fig    

        self.xlim, self.ylim = self.__get_axis_limits()

        # add slider
        fig.subplots_adjust(top=0.9)
        axis_window = fig.add_axes([0.1, 0.95, 0.8, 0.03])
        self.slider = widgets.Slider(
            axis_window, 'Time Window', 
            0.5, 8, valinit=0.5, valstep=0.5,
            valfmt="%g hour(s)",
            orientation='horizontal',
            initcolor='none'
        )
        self.slider.on_changed(self.on_slider_change)

        fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        self.__update(0, False)
        plt.show()
    
    def __update(self, new_mean: int, do_redraw = True):
        if new_mean == self.current_mean:
            return
        self.current_mean = new_mean
        
        if do_redraw:
            self.axes.clear()
        
        self.df.plot.scatter(
            ax=self.axes, 
            x='sleep_length', y=f"mean_{new_mean}",
            xlim=self.xlim, ylim=self.ylim,
            # :g ensures no trailing zeroes are there. 1.0 -> 1
            ylabel='Average Steps',
            xlabel='Sleep Duration',
            title=self.__get_title(),
        )

        self.axes.xaxis.set_major_locator(MinuteLocator(byminute=[0, 30]))
        self.axes.xaxis.set_major_formatter(DateFormatter('%H:%Mh'))
        self.axes.xaxis.set_minor_locator(AutoMinorLocator(2))

        self.fig.canvas.draw_idle()

    def __get_title(self):
        hours = self.current_mean * 0.5 + 0.5

        inserted_label = f"{hours:g} Hour"

        if hours < 1:
            inserted_label = f"{int(hours * 60)} Minutes"
        elif hours != 1:
            inserted_label += 's'
        return f"Average Steps {inserted_label} After Waking Up"       

    def on_slider_change(self, val):
        actual_val = int(val * 2) - 1
        self.__update(actual_val)
    
    def on_key_press(self, event):
        # Slider.set_val() triggers the Slider.on_change event.
        val = self.slider.val
        valmin = self.slider.valmin
        valmax = self.slider.valmax
        step = self.slider.valstep

        if event.key == 'right' and val <= (valmax - step):
            self.slider.set_val(val + step)
        elif event.key == 'left' and val >= (valmin + step):
            self.slider.set_val(val - step)


if __name__ == '__main__':
    csv_merger.generate_data()
    # Even though this variable seems unnecessary, it prevents
    # Python from garbage collecting our graph.
    # Do not remove this assignment.
    fig = ScatterFigure()
    fig.plot()