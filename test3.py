from pandas import read_excel
from openpyxl.utils.dataframe import dataframe_to_rows
import statistics

dataframe = read_excel('SampleData.xlsx', engine='openpyxl')

data = []
for row in dataframe_to_rows(dataframe, index=False, header=False):
    index = 0
    for column in row:
        if len(data) < index + 1:
            data.append([])

        data[index].append(column)
        index += 1
print(data)

print('\n To sum up all of the elements in a given dimension that are summable')

# We sum columns that can be summed up using map()

mylist_units = dataframe['Units'].values.tolist()
mylist_unitscost = dataframe['Unit Cost'].values.tolist()
mylist_total = dataframe['Total'].values.tolist()

print('\n Sum of units')
print(sum(map(int, mylist_units)))

print('\n Sum of Unit Cost')
print(sum(map(int, mylist_unitscost)))

print('\n Sum of Total')
print(sum(map(int, mylist_total)))

print('\n To average  all of the elements in a given dimension that are summable\n')

# We average columns that can be summed up using map()

mylist_units = dataframe['Units'].values.tolist()


# print(statistics.mean(mylist_units))

def Average(l):
    avg = sum(l) / len(l)
    return avg


average = Average(mylist_units)
average_unitcost = Average(mylist_unitscost)
average_total = Average(mylist_total)

print("Average of Units is", average)
print("Average of unit cost is", average_unitcost)
print("Average of Total is", average_total)

#  create functions to support shrink and expand of array size dynamically

