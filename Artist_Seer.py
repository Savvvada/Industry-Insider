import numpy as np
import csv
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from csv import writer
import shutil

# A list of different variables essential for this module
csvhldr = None # A variable that will be set equal to the path of the csv file being passed, it should be a string.
name = ""
songTitle = ""
holder = []


def assign(a,b): # This function assigns user entered values to the variables "name" and "songTitle"
    global name
    global songTitle
    name = a
    songTitle = b
    look_for_artist()

def look_for_artist():
    global name
    present = False

    feature = ['artist']
    df = pd.read_csv("toMerge/chart2000-albummonth-0-3-0060.csv", usecols = feature)
    for index, row in df.iterrows():
        if name == row['artist']:
            present = True
            break
    if not present:
        print('ERROR: There is no empirical data for the artist whose name you\'ve entered ')

def csv_processor():
    global holder
    count = range(4)
    n = 0
    for c in range(1):
        n = n + 1
        print('genre' + str(n))
        file1 = pd.read_csv("inputting/input1.csv")
        file1.dropna(inplace = True)
        file1['artist'] = file1['artist'].astype(str)
        file2 = pd.read_csv('genre/genre' + str(n) + '.csv')
        file2['name'] = file2['name'].astype(str)
        file2['genre'] = file2['genre'].astype(str)
        file1["artist"] = file1["artist"]
        file2["name"] = file2["name"]
        file2["genre"] = file2["genre"]
        z = 0
        w = 0
        for x in file1.iloc:
            for y in file2.iloc:
                #print(x['artist'])
                #print(y['name'])
                if x['artist'] == y['name']:
                    w = w + 1
                    print(x['artist'])
                    print(y['name'])
                    holder[z] = (y['genre'])
                    print(str(holder[z]))
            z = z + 1
        print(w)
    shutil.copyfile('inputting/input1.csv', 'outputting/output3.csv')
    file3 = pd.read_csv("outputting/output3.csv")
    file3['genre'] = holder
    file3.to_csv("output4.csv", index=True)


def prequisites():
    global holder
    holder = np.empty( (len(pd.read_csv("inputting/input1.csv"))) , dtype = object)

if __name__ == '__main__':
    prequisites()
    csv_processor()