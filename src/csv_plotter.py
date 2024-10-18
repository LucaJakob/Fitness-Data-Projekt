import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path 
from csv_merger import file_merger_days

if __name__ == '__main__':
    VIEW_CSV = '1.csv'

    merged_dir = Path('./data/merged')

    # csv merger has not executed yet
    if not merged_dir.is_dir():
        file_merger_days()
    
    steps = "monitoring.steps[steps]"

    df = pd.read_csv(merged_dir / VIEW_CSV)
    df[["monitoring.timestamp[s]"]].plot() 
    plt.show()

