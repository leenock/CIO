import openpyxl
from pathlib import Path
import pandas as pd

df_samples = pd.read_excel(r'SampleData.xlsx', engine='openpyxl')
print(df_samples) # print all the records


orderDate = df_samples['OrderDate'].values.tolist()
Region = df_samples['Region'].values.tolist()
Item = df_samples['Item'].values.tolist()
units = df_samples['Units'].values.tolist()
Unit_Cost = df_samples['Unit Cost'].values.tolist()
Total = df_samples['Total'].values.tolist()

count = 0

while (count < len(units)):
    print("orderDate: " + str((orderDate[count])) + " " + "Region: " + Region[count] + " " + " " + "Item: " + Item[
        count] +
          "units: " + str(units[count]) + " " + "Unit_Cost: " + str(Unit_Cost[count]) + "Total: " + str(Total[count]))
    count = count + 1
