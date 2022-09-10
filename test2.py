import openpyxl
from pathlib import Path
import pandas as pd
from pandas.io import json
from pip._internal.utils.misc import tabulate

df_samples = pd.read_excel(r'FinancialSample.xlsx', engine='openpyxl')
print(df_samples.head())

df_sample_x = df_samples.groupby("Country").mean()
df_sample_x = df_sample_x.drop(['Month Number', 'Year'], axis=1)
print((df_sample_x))

df_sample_x = df_samples.groupby("Country").sum()
df_sample_x = df_sample_x.drop(['Month Number', 'Year'], axis=1)
print((df_sample_x))