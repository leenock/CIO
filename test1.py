import pandas as pd
from pyexcel._compact import OrderedDict
import csv

OrderDate = []
Region = []
Rep = []
Item = []
Units = []
Unit_Cost = []
Total = []

with open('SampleData.csv', 'r') as file:
    sample_ = csv.reader(file)

df_samples = pd.read_excel(r'SampleData.xlsx', engine='openpyxl')
#print(df_samples) # print all the records

for row in df_samples:
        OrderDate.append(row[0])
        Region.append(row[1])
        Rep.append(row[2])
        Item.append(row[3])
        Units.append(row[4])
        Unit_Cost.append(row[5])
        Total.append(row[6])

print(OrderDate)
print(Region)
print(Rep)
print(Item)
print(Units)
print(Unit_Cost)
print(Total)

# We sum columns that can be summed up using map()

