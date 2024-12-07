import pandas as pd
from utils import read_all_csv

def load_hrv(final_df: pd.DataFrame, time_col: str):
    for i in range(1, 32):
        all_df = read_all_csv(f"data/2024-08/CSV/CSV_DATA/{i}/HRV/*.csv")
        for df in all_df:
            parse_hrv(final_df, time_col, df)

def parse_hrv(final_df: pd.DataFrame, time_col: str, df: pd.DataFrame):
    hrv_ts = 'hrv_value.timestamp'
    columns = ['hrv_value.value[ms]']

    



if __name__ == "__main__":
    print("Hello")

# stress_level.stress_level_value
# respiration_rate.respiration_rate[breaths/min]
# monitoring.heart_rate[bpm]
# monitoring.steps[steps]
# respiration_rate.timestamp$



#Time series google
# ARIMA-Modell
# Multi-Variablen-Modell
