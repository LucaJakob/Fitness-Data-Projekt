import pandas as pd
import matplotlib.pyplot as plt

steps = "monitorning.steps[steps]"


df = pd.read_csv(r'C:\Users\silva\Documents\CDS\DS\Projektarbeit_Gesundheitsdaten\Daten\August_24_Copy\CSV\MERGED_3\FullMonth.csv')

df_clean = df.drop_duplicates(subset=['monitoring.timestamp[s]'])


df[["monitoring.timestamp[s]"]].plot()

plt.show()