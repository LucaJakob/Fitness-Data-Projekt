import os
import pandas as pd
import glob


def file_merger_days():
    
    folder = 1

    for i in range(0,31):
    

        all_files = glob.glob(rf'C:\Users\silva\Documents\CDS\DS\Fitness-Data-Projekt\data\2024-08\CSV\CSV_DATA\{folder}\WELLNESS/*.csv',recursive=True)
        all_df = []
        for f in all_files:
            df = pd.read_csv(f, sep=',')
            all_df.append(df)

        df=all_df[0]
        for df_temp in all_df[1:]:
            df = pd.concat([df, df_temp], ignore_index=True, sort=False)
        #removing duplicate monitoring.timestamp[s] due to tracking error
        df_clean = df.drop_duplicates(subset=['monitoring.timestamp[s]'])
        df_clean.to_csv(rf'C:\Users\silva\Documents\CDS\DS\Fitness-Data-Projekt\data\merged\{folder}.csv',sep=',')
        folder += 1



file_merger_days()
