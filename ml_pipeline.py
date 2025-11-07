import pandas as pd

matches_df = pd.read_csv('Raw_Data_Kaggle/deliveries.csv')
deliveries_df = pd.read_csv('Raw_Data_Kaggle/deliveries.csv')

print(matches_df.head())
print(deliveries_df.head())