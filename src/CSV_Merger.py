import pandas as pd
import glob
from pathlib import Path

def file_merger_days():
    
    path = Path('./data/2024-08/CSV/CSV_DATA')
    output_dir = Path('./data/merged')

    for i in range(1,32):
        
        all_files = glob.glob(rf'{path}/{i}/WELLNESS/*.csv',recursive=True)
        all_df = []
        for f in all_files:
            df = pd.read_csv(f, sep=',')
            all_df.append(df)

        df=all_df[0]
        for df_temp in all_df[1:]:
            df = pd.concat([df, df_temp], ignore_index=True, sort=False)
        #removing duplicate monitoring.timestamp[s] due to tracking error
        df_clean = df.drop_duplicates(subset=['monitoring.timestamp[s]'])
        df_clean.to_csv(rf'{output_dir}/{i}.csv',sep=',')


if __name__ == '__main__':
    file_merger_days()