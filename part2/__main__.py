import pandas as pd
from utils import read_all_csv
from SleepState import SleepState
import sklearn.ensemble as skensemble
import sklearn.tree as sktree
import sklearn.metrics as skmetrics
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_hrv(train_df: pd.DataFrame, test_df: pd.DataFrame, time_col: str):
    output = train_df
    test_data = test_df
    for i in range(1, 32):
        output = read_hrv(f"data/2024-08/CSV/CSV_DATA/{i}/HRV/*.csv", output, time_col)
        test_data = read_hrv(f"data/2024-09/CSV/CSV_DATA/{i}/HRV/*.csv", test_data, time_col)
    return (output, test_data)

def read_hrv(path: str, final_df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    output = final_df
    hrv_ts = 'hrv_value.timestamp'
    columns = ['hrv_value.value[ms]'] # Herzfrequenzvariabilität
    all_df = read_all_csv(path, index_col=hrv_ts)
    for df in all_df:
        if columns not in df.columns.values:
            continue
        new_df = df[columns]
        new_df = new_df.dropna(subset=columns)
        new_df.rename(index={hrv_ts: time_col})
        output = pd.concat([output, new_df])
    return output

def load_monitoring(train_df: pd.DataFrame, test_df: pd.DataFrame, time_col: str):
    output = train_df
    test_data = test_df
    for i in range(1, 32):
        output = read_monitoring(f"data/2024-08/CSV/CSV_DATA/{i}/WELLNESS/*.csv", output, time_col)
        test_data = read_monitoring(f"data/2024-09/CSV/CSV_DATA/{i}/WELLNESS/*.csv", test_data, time_col)
    return (output, test_data)

def read_monitoring(path: str, final_df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    output = final_df
    monit_ts = 'monitoring.timestamp[s]'

    all_df = read_all_csv(path, index_col=monit_ts)
    for df in all_df:
        columns = ['monitoring.heart_rate[bpm]','monitoring.steps[steps]']
        if any([x for x in columns if x not in df.columns.values]):
            continue
        new_df = df[columns]
        new_df = new_df.dropna(subset=columns)
        new_df.rename(index={monit_ts: time_col}, columns={columns[0]: "bpm", columns[1]: "steps"})
        output = pd.concat([output, new_df])
    return output




def load_breathing(train_df: pd.DataFrame, test_df: pd.DataFrame, time_col: str):
    output = train_df
    test_data = test_df
    for i in range(1, 32):
        output = read_breathing(f"data/2024-08/CSV/CSV_DATA/{i}/WELLNESS/*.csv", output, time_col)
        test_data = read_breathing(f"data/2024-09/CSV/CSV_DATA/{i}/WELLNESS/*.csv", test_data, time_col)
    return (output, test_data)

def read_breathing(path: str, final_df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    output = final_df
    breath_ts = 'respiration_rate.timestamp'
    all_df = read_all_csv(path, index_col=breath_ts)
    for df in all_df:
        columns = ['respiration_rate.respiration_rate[breaths/min]']
        if columns not in df.columns.values:
            continue
        new_df = df[columns]
        new_df = new_df.dropna(subset=columns)
        new_df.rename(index={breath_ts: time_col}, columns={columns[0]: "breaths/min"})
        output = pd.concat([output, new_df])
    return output

# Beispiel: Schlafphasen über die Zeit
def plot_sleep_phases(df, sleep_level_col):
    """
    Plot der Schlafphasen über eine Zeitspanne, wenn Zeitstempel im Index sind.
    
    Args:
        df: Pandas DataFrame mit Zeitstempeln im Index und Schlafphasen.
        sleep_level_col: Spaltenname der Schlafphasen.
    """
    # Sicherstellen, dass der Index datetime ist
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df.index = pd.to_datetime(df.index, unit='s')
    
    # Mapping für lesbare Schlafphasen
    sleep_labels = {1: "Awake", 2: "Light sleep", 3: "Deep sleep", 4: "REM"}
    df['sleep_phase_label'] = df[sleep_level_col].map(sleep_labels)
    
    # Plot erstellen
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df[sleep_level_col], marker='o', linestyle='-', alpha=0.7)
    plt.yticks(list(sleep_labels.keys()), sleep_labels.values())
    plt.xlabel("Zeit")
    plt.ylabel("Schlafphasen")
    plt.title("Schlafphasen über die Zeit")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_sleep_phases_one_day(df, sleep_level_col, day):
    """
    Plot der Schlafphasen für einen spezifischen Tag, wenn der Timestamp im Index ist.
    
    Args:
        df: Pandas DataFrame mit Zeitstempeln im Index und Schlafphasen.
        sleep_level_col: Spaltenname der Schlafphasen.
        day: Das Datum als String im Format "YYYY-MM-DD".
    """
    # Sicherstellen, dass der Index datetime ist
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df.index = pd.to_datetime(df.index, unit='s')
    
    # Filtere den DataFrame für den angegebenen Tag
    start_time = pd.to_datetime(day)
    end_time = start_time + pd.Timedelta(days=1)
    df_filtered = df[(df.index >= start_time) & (df.index < end_time)]
    
    if df_filtered.empty:
        print(f"Keine Daten für den Tag {day} vorhanden.")
        return
    
    # Mapping für lesbare Schlafphasen
    sleep_labels = {1: "Awake", 2: "Light sleep", 3: "Deep sleep", 4: "REM"}

    df_filtered = df_filtered.copy()  # Erstelle eine Kopie, um sicherzustellen, dass keine View verwendet wird
    df_filtered['sleep_phase_label'] = df_filtered[sleep_level_col].map(sleep_labels)
    
    # Plot erstellen
    plt.figure(figsize=(12, 6))
    plt.plot(df_filtered.index, df_filtered[sleep_level_col], marker='o', linestyle='-', alpha=0.7)
    plt.yticks(list(sleep_labels.keys()), sleep_labels.values())
    plt.xlabel("Zeit")
    plt.ylabel("Schlafphasen")
    plt.title(f"Schlafphasen für den Tag {day}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Funktion zum Entfernen von Tages-Wachphasen
def filter_daytime_wake_phases(df, sleep_level_col, wake_level=1, hour_threshold=1) -> pd.DataFrame:
    """
    Entfernt Wachphasen während des Tages basierend auf kontinuierlichen Phasen.
    Behandelt den Übergang am Morgen und Abend entsprechend den Bedingungen.
    """
    # Konvertiere den Index (Unix-Timestamps) zu datetime, falls nötig
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df.index = pd.to_datetime(df.index, unit='s')

    # Sortiere nach Index
    df = df.sort_index()

    # Identifiziere Wachphasen (sleep_level == wake_level)
    df['is_wake'] = (df[sleep_level_col] == wake_level)

    # Berechne kontinuierliche Wachzeiten
    df['wake_duration'] = (
        df['is_wake']
        .astype(int)
        .groupby((df['is_wake'] != df['is_wake'].shift()).cumsum())
        .cumsum()
    )

    # Entferne Tageswachphasen
    filtered_indices = []
    start_wake_time = None
    for i, (timestamp, row) in enumerate(df.iterrows()):
        if row['is_wake']:
            if start_wake_time is None:
                start_wake_time = timestamp
            if (timestamp - start_wake_time).total_seconds() > hour_threshold * 3600:
                # Ab hier Tages-Wachphasen entfernen
                filtered_indices.append(i)
        else:
            start_wake_time = None

    # Rückwirkend 1 Stunde Wachphasen am Abend behalten
    for i in range(len(df) - 1, -1, -1):
        if df.iloc[i][sleep_level_col] != wake_level:  # Start Nachtphase gefunden
            end_sleep_time = df.index[i]
            for j in range(i, -1, -1):
                if (df.index[j] - end_sleep_time).total_seconds() > -hour_threshold * 3600:
                    break
                filtered_indices = list(set(filtered_indices) - {j})
            break

    # Entferne gefilterte Zeilen
    df = df.drop(df.index[filtered_indices]).drop(columns=['is_wake', 'wake_duration'])

    return df

def map_index(df: pd.DataFrame):
    dates = pd.DatetimeIndex(df.index.values)
    # see https://stackoverflow.com/questions/16453644/regression-with-date-variable-using-scikit-learn
    df['day'] = dates.day
    df['hour'] = dates.hour
    df['minute'] = dates.minute
    df['second'] = dates.second

# enum:
# Awake: 1
# Light sleep: 2
# Deep sleep: 3
# REM: 4

if __name__ == "__main__":
    ts_col = "timestamp"
    print("Loading HRV data ...")
    df, test_df = load_hrv(pd.DataFrame(), pd.DataFrame(), ts_col)

    print("Loading monitoring data ...")
    df, test_df = load_monitoring(df, test_df, ts_col)

    print("Loading breathing data ...")
    df, test_df = load_breathing(df, test_df, ts_col)

    print("Loading sleep data ...")
    df['sleep_level'] = df.index.map(lambda x: SleepState.get(x, mode="train"))
    test_df['sleep_level'] = test_df.index.map(lambda x: SleepState.get(x, mode="test"))

    wake_count  = (df['sleep_level'] == 1).sum()
    sleep_count = (df['sleep_level'] == 2).sum()

    # plot_sleep_phases_one_day(test_df, sleep_level_col="sleep_level", day="2004-09-29")

    print("Filtering daytime wake phases...")
    df = filter_daytime_wake_phases(df, sleep_level_col='sleep_level')
    test_df = filter_daytime_wake_phases(test_df, sleep_level_col='sleep_level')

    #plot_sleep_phases(df, sleep_level_col="sleep_level")
    #plot_sleep_phases_one_day(test_df, sleep_level_col="sleep_level", day="2004-09-29")

    # TODO: Figure out if timestamps should be variable in rows or index is fine
    map_index(df)
    map_index(test_df)
    target_vals = df['sleep_level']
    df.drop(columns=['sleep_level'], inplace=True)

    test_vals = test_df['sleep_level']
    test_df.drop(columns=['sleep_level'], inplace=True)
    print("Data loaded")
    print("Dataset:", df.shape)
    print("Test dataset:", test_df.shape)

    print("Training Models ...")
    models = {
        'Random Forest': skensemble.RandomForestClassifier(random_state=42),
        'Decision Tree': sktree.DecisionTreeClassifier(random_state=42),
        'Histogram-Based Gradient Boosting': skensemble.HistGradientBoostingClassifier(random_state=42),
    }

    results = []
    for model_name, model in models.items():
        # Modell trainieren
        print(f"Training {model_name}")
        model.fit(df.values, target_vals.values)
        print(f"Predicting with {model_name}")
        # Vorhersagen treffen
        y_pred = model.predict(test_df)
        
        # Metriken berechnen
        accuracy = skmetrics.accuracy_score(test_vals, y_pred)
        f1 = skmetrics.f1_score(test_vals, y_pred, average='weighted')
        conf_matrix = skmetrics.confusion_matrix(test_vals, y_pred)
        
        # Ergebnisse speichern
        results.append({
            'Model': model_name,
            'Accuracy': accuracy,
            'F1 Score': f1,
            'Confusion Matrix': conf_matrix
        })

    print("Eindeutige Werte in den Vorhersagen:", np.unique(y_pred))

    results_df = pd.DataFrame(results).drop(columns=['Confusion Matrix'])
    print(results_df)

    enum_labels = {1: "Awake", 2: "Light sleep", 3: "Deep sleep", 4: "REM"}



for result in results:
    plt.figure(figsize=(6, 4))
    conf_matrix = result['Confusion Matrix']

    # Mögliche Klassen aus dem Enum definieren
    enum_classes = [1, 2, 3, 4]  # Werte aus dem Enum
    tick_labels = [enum_labels.get(i, str(i)) for i in enum_classes]

    # Heatmap erstellen
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=tick_labels, yticklabels=tick_labels)
    plt.title(f"Confusion Matrix: {result['Model']}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

# Time series google
# Random Forest
# ANN
# Decision Tree Classifier
# Logistic Regression
# Naive Bayes