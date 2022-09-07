import openpyxl
from pathlib import Path
import pandas as pd
from pandas.io import json

df_samples = pd.read_excel(r'SampleData.xlsx', engine='openpyxl')
for key, item in df_samples.items():
    print(json.dumps({key: item}))
