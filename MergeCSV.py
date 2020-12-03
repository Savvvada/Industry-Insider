import numpy as np
import csv
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import shutil
import os, os.path
from csv import reader, writer

colName = [' ', 'month', 'position', 'artist', 'album', 'indicativerevenue', 'us', 'uk', 'de', 'fr', 'ca', 'au', 'genre']
input_group = []
checkboxes = []

def prerequisite():
    list = os.listdir('inputting/')  # dir is your directory path
    number_files = len(list)
    for x in range(0, number_files):
        y = x + 1
        x = globals()['input%s' % y] = pd.read_csv("inputting/input" + str(y) + ".csv")
        input_group.append(x)
    for x in range(0, number_files):
        for y in range(0, len(colName)):
            if colName[y] in input_group[x].columns:
                # a = (input_group[x][colName[y]])
                checkboxes.append(input_group[x][colName[y]])
            elif y == 0 :
                store = []
                for i in range(len(pd.read_csv("inputting/input1.csv"))):
                    store.append(str(1))
                checkboxes.append(store)
    with open('outputting/output.csv', 'w') as my_csv:
        for x in range(1):
                csvWriter = writer(my_csv, delimiter=',')
                csvWriter.writerows(checkboxes)

    csv = pd.read_csv("outputting/output.csv")
    df_csv = pd.DataFrame(data=csv)
    transposed_csv = df_csv.T
    df_csv.insert(0, 'new-col', '1')
    transposed_csv.to_csv(r'outputting/output2.csv',header=['0','1','2','3','4','5','6','7','8','9','10','11'], index=False)




if __name__ == '__main__':
    prerequisite()

