import numpy as np
import csv
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import tree, metrics
import graphviz

dataset = None

def load_csv():
    global dataset

    dataset = pd.read_csv('outputting/output2.csv')
    dataset = dataset.drop_duplicates(subset='3')
    dataset = dataset[dataset['11'].notna()]
    op = pd.get_dummies(dataset['2'])
    op.to_csv('outputting/output6.csv')
    op = pd.get_dummies(dataset['11'])
    op.to_csv('outputting/output7.csv')
    dataset2 = pd.read_csv('outputting/output7.csv')
    dataset3 = pd.read_csv('outputting/output6.csv')
    dataset = dataset.drop(['11','2'], axis=1)
    dataset.reset_index(drop=True, inplace=True)
    dataset = pd.concat([dataset, dataset2], axis=1, sort=False)
    dataset = dataset.drop(['Unnamed: 0'], axis=1) #The index keeps getting added as a column, so I routinely drop it on Concatenation
    dataset = pd.concat([dataset, dataset3], axis=1, sort=False)
    dataset = dataset.drop(['Unnamed: 0'], axis=1)

    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']#I use this array just to concatenate with the iterative values so I have to type less
    dates = []#This will contain the dates present in the csv file which I wish to replace
    datesEncoded = []#This will be equal to the values I want to substitute for the dates, it's just an incremental value I need to capture the time-based cyclical nature of the data
    encodedCount = 0 #This is the incremental count that will go in "datesEncoded", it's so that the model can make infrences about values past a certain date. If I entered dates as binary categories like I did with artist names, then I couldn't make valid infrences aabout future dates.
    for i in range(22):# I want the model to predict a year in advance, hence I chose 21, being that it's December 2020 at the time of writing this.
        if i < 10:
            g = '200'+ str(i)
        else:
            g = '20'+ str(i)
        for j in range(12):
            dates.append(months[j]+g)
            encodedCount += 1
            datesEncoded.append(encodedCount)
    print(dates)
    print(datesEncoded)

    for i in range(5,11): # I get the average ranking for each ranking column and filter out the missing values "-" with the average.
        mean3 = dataset[str(i)]
        indexNames = dataset[(dataset[str(i)] == '-')].index
        mean3.drop(indexNames, inplace=True)
        mean3 = mean3.astype(int)
        mean2 = mean3.mean()
        dataset[str(i)] = dataset[str(i)].str.replace("-", str(mean2))
        print(mean2)

    dataset['0'] = dataset['0'].str.replace(" ", "")
    dataset['0'] = dataset['0'].replace(dates, datesEncoded ) #Here I use the array of dates I created in the loop as the 'old values' to be replaced with their encoded equivalent
    dataset['0'] = dataset['0'].astype(int)
    # dataset['0'].to_csv('encodedDate.csv') #uncommenting this will produce a csv that shows how the dates have been encoded
    dataset = dataset.drop(['1','3'], axis=1) #Albeit I created multiple categorical columns for "artist" and "genre", I hadn't made one for album name as I don't see it to hold much relevance over the content of the album themselves and therefore drop it in this line along with indexes created in earlier steps of the program, however one could use the album names and ascribe significance to them in one regard or another such as "phonetic appeal, marketability, etc" , it'd be inherently bias and inconsistent.

    dataset.to_csv('ProcessedData.csv')


def choose_x_y():
    global dataset

    x = dataset.drop(['4'], axis=1)
    y = dataset['4']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=2) # I split up  the data so it can test on data it hasn't seen
    dtr1 = RandomForestRegressor(n_estimators=360, min_samples_leaf=3, min_samples_split =9, max_features=.95, n_jobs=-1, oob_score=True) #The more trees in the forest the better, reduces the chance of anomalies. I noticed very little changes in accuracy beyong 360 trees.
    dtr1.fit(x_train, y_train) #I put in the training data, to test for the unseen test data
    # dtr1.fit(x, y) # One would uncomment this line, and comment the one above when they want to test with already seen data.


    # x_test is passed into the predict function to predict all values for the "y_test" and see if y_pred matches y_test.
    # x_test has a size of 729
    predictions = dtr1.predict(x_test)
    hit_Mark = 0
    off_Mark = 0
    Total = len(predictions)
    y_test = y_test.to_numpy(y_test)
    print(y_test)
    # print(predictions)
    for i in range(len(predictions)):
    #Here is where I introduce the error margin of 11%,
        if (predictions[i] <= y_test[i] + (y_test[i] * marginerror)) & (predictions[i] >= (y_test[i] - (y_test[i] * marginerror))):
            hit_Mark += 1
            plt.plot(y_test[i], predictions[i], color='green', marker='o')
        if not((predictions[i] <= y_test[i] + (y_test[i] * marginerror)) & (predictions[i] >= (y_test[i] - (y_test[i] * marginerror)))):
            off_Mark += 1
            plt.plot(y_test[i], predictions[i], color='red', marker='x')


    print('Accuracy is : ' + str(hit_Mark/Total))
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, predictions))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, predictions))
    # print(predictions)
    # plt.scatter(y_test, predictions)
    plt.xlabel("Actual album revenue")
    plt.ylabel("Predicted album revenue")
    plt.show()



if __name__ == '__main__':
    global in_val
    global marginerror
    global see

    in_val = input("Would you like to see the performance of the model via: Actual vs Predicted Scatter Plot, Accuracy, Mean"
          " Absolute error, and Mean Squared Error? If so enter \"1\" :   ")

    if in_val == str(1):
        marginerror = input('please choose the margin of error you\'d like to introduce, type \"15\" for 15% type \"7\" for 7% etc:    ')
        marginerror = float(str('.'+marginerror))
        load_csv() # The preprocessing happens here
        choose_x_y() # This feeds the processed data to the model

