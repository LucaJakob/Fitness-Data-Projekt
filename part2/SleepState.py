import pandas as pd
from utils import read_all_csv

class SleepState:
    __loaded = False
    __train_limits: dict[int, tuple[int, int]] = {}
    __test_limits: dict[int, tuple[int, int]] = {}
    __train_map: dict[int, int] = {}
    __test_map: dict[int, int] = {}

    @staticmethod
    def __load(training_month="2024-08", testing_month="2024-09"):
        """
        Load sleep data for training and testing periods separately.
        """
        # Trainingsdaten laden
        print("Lade Trainingsdaten...")
        for day in range(1, 32):  # Tage im Trainingsmonat
            all_df = read_all_csv(f"data/{training_month}/CSV/CSV_DATA/{day}/SLEEP/*.csv")
            if len(all_df) == 0:
                continue
            SleepState.__parse_df(all_df, day, "train")

        # Testdaten laden
        print("Lade Testdaten...")
        for day in range(1, 32):  # Tage im Testmonat
            all_df = read_all_csv(f"data/{testing_month}/CSV/CSV_DATA/{day}/SLEEP/*.csv")
            if len(all_df) == 0:
                continue
            SleepState.__parse_df(all_df, day, "test")

        SleepState.__loaded = True

    @staticmethod
    def __parse_df(df_list: list[pd.DataFrame], day: int, mode: str):
        """
        Parse the provided list of dataframes and load them into the mapper.
        Updates train or test limits and maps based on the mode.
        """
        col_level = 'sleep_level.sleep_level'
        col_time = 'sleep_level.timestamp[s]'

        for df in df_list:
            df = df[(df[col_level].notna()) & (df[col_time].notna())]
            df = df.sort_values(by=col_time)

            if not df.empty:
                first_timestamp = df.iloc[0][col_time]
                last_timestamp = df.iloc[-1][col_time]

                if mode == "train":
                    SleepState.__train_limits[day] = (first_timestamp, last_timestamp)
                elif mode == "test":
                    SleepState.__test_limits[day] = (first_timestamp, last_timestamp)

            for _, row in df.iterrows():
                timestamp = row[col_time]
                level = int(row[col_level])

                if mode == "train":
                    if SleepState.__train_map.get(timestamp) is None:
                        SleepState.__train_map[timestamp] = level
                elif mode == "test":
                    if SleepState.__test_map.get(timestamp) is None:
                        SleepState.__test_map[timestamp] = level

    @staticmethod
    def get(timestamp: int, mode: str = "train") -> int:
        """
        Get the sleep level for the given timestamp based on the mode (train or test).
        Returns:
            1: Awake
            2: Light Sleep
            3: Deep Sleep
            4: REM
        """
        if not SleepState.__loaded:
            SleepState.__load()

        limits = SleepState.__train_limits if mode == "train" else SleepState.__test_limits
        sleep_map = SleepState.__train_map if mode == "train" else SleepState.__test_map

        # Prüfen, ob der Zeitstempel innerhalb der Schlafgrenzen liegt
        for day, (start, stop) in limits.items():
            if start <= timestamp <= stop:
                break
        else:
            return 1  # Wach

        # Suche den letzten bekannten Schlaflevel
        last_level = 1  # Standardwert: Wach
        for ts, level in sorted(sleep_map.items()):
            if ts > timestamp:
                break
            last_level = level

        return last_level

    @staticmethod
    def is_during_sleep(timestamp: int, mode: str = "train") -> bool:
        """
        Determines whether a provided timestamp falls within a valid sleep period.
        """
        limits = SleepState.__train_limits if mode == "train" else SleepState.__test_limits
        for start, stop in limits.values():
            if start <= timestamp <= stop:
                return True
        return False
