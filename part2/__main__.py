import pandas as pd
from utils import read_all_csv
from SleepState import SleepState
import sklearn.ensemble as skensemble
import sklearn.tree as sktree
import sklearn.metrics as skmetrics
import matplotlib.pyplot as plt
import seaborn as sns

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
    df['sleep_level'] = df.index.map(lambda x: SleepState.get(x))
    test_df['sleep_level'] = test_df.index.map(lambda x: SleepState.get(x))

    # TODO: Figure out if timestamps should be variable in rows or index is fine
    # df['time_posix'] = df.index 
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
        model.fit(df, target_vals)
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

    results_df = pd.DataFrame(results).drop(columns=['Confusion Matrix'])
    print(results_df)

    for result in results:
        plt.figure(figsize=(6, 4))
        sns.heatmap(result['Confusion Matrix'], annot=True, fmt='d', cmap='Blues')
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