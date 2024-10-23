import pandas as pd
import glob
from pathlib import Path
import shutil
from csv_columns import WellnessColumns, SleepColumns, StaticData

class MetaData:
    def __init__(self, 
        shape: tuple[int, int] = (0, 0), 
        removed_rows    = 0, 
        removed_columns = 0
    ):
        self.rows, self.columns = shape
        self.removed_rows = removed_rows
        self.removed_columns = removed_columns
    
    def __str__(self):
        return f"{self.rows} (- {self.removed_rows}) Rows x {self.columns} (- {self.removed_columns}) Columns"

def merge_df(
    regex_path: str, 
    drop_duplicate_cols = [], 
    drop_cols = []
):
    """
Merge multiple CSV files into one pandas DataFrame.

Parameters
----------

regex_path : str
    New labels / index to conform the axis specified by 'axis' to.
drop_duplicate_cols : column label or sequence of labels, optional
    Drop duplicate rows based on this subset of columns. If omitted, no rows are dropped.
drop_cols : single label or iterable or list of iterables, optional
    Drop the specified columns. No columns are dropped by default.

Returns
-------
A tuple containing two items:
- DataFrame. If the specified filepath has no CSV files, an empty DataFrame is returned.
- MetaData containing information about the merge
    """
    
    all_files = glob.glob(regex_path,recursive=True)
    all_df = [pd.read_csv(f, sep=',') for f in all_files]

    removed_rows = 0
    removed_cols = 0

    # No data is present
    if len(all_df) == 0:
        return (pd.DataFrame(), MetaData())

    df = pd.concat(all_df, ignore_index=True, sort=False)

    if len(drop_duplicate_cols) > 0:
        old_rows = df.shape[0]
        df = df.drop_duplicates(subset=drop_duplicate_cols)
        removed_rows = old_rows - df.shape[0]

    if len(drop_cols) > 0:
        for col in drop_cols:
            # only count dropped columns if they actually exist
            try:
                df = df.drop(col, axis=1, errors='raise')
                # col may be a list-like
                removed_cols += 1 if isinstance(col, str) else len(col)
            except KeyError:
                continue
            
    return (df, MetaData(df.shape, removed_rows, removed_cols))

def file_merger_wellness(output_dir: Path):
    path = Path('./data/2024-08/CSV/CSV_DATA')
    output_wellness = output_dir / 'wellness'

    output_wellness.mkdir(parents=True, exist_ok=True)

    for i in range(1,32):
        df, meta = merge_df(
            rf'{path}/{i}/WELLNESS/*.csv', 
            drop_duplicate_cols=[WellnessColumns.monitoring.timestamp_s],
            drop_cols=StaticData()
        )
        
        if df.empty:
            continue

        df.to_csv(rf'{output_wellness}/{i}.csv',sep=',')
        print(f"wellness/{i}.csv\t", meta)

def file_merger_sleep(output_dir: Path):
    
    path = Path('./data/2024-08/CSV/CSV_DATA')
    output_sleep = output_dir / 'sleep'

    
    output_sleep.mkdir(parents=True, exist_ok=True)

    for i in range(1,32):
        df, meta = merge_df(
            rf'{path}/{i}/SLEEP/*.csv', 
            drop_duplicate_cols=[SleepColumns.assessment.timestamp_s], 
            drop_cols=StaticData()
        )
        
        if df.empty:
            continue

        df.to_csv(rf'{output_sleep}/{i}.csv',sep=',')
        print(f"sleep\\{i}.csv\t", meta)


def generate_data(output_dir: Path):
    print('\n----Wellness Data----')
    file_merger_wellness(output_dir)
    print('\n----Sleep Data----')
    file_merger_sleep(output_dir)

if __name__ == '__main__':
    merged_path = Path('./data/merged')

    print('Cleaning ./data/merged directory ...')
    shutil.rmtree(merged_path)

    print('Generating CSV data ...')
    generate_data(merged_path)
    print('Merged CSV data created.')