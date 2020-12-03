import numpy as np
import csv
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

regressor = DecisionTreeRegressor(random_state = 0)  # create a regressor object

def load_csv():

    ##e = LabelEncoder()
    dataset = pd.read_csv('outputting/output2.csv')
   # dataset = pd.read_csv(csvhldr) #This would be used when a string is actually sent to be the place holder value

def load_n_train_csv():

    dtc = DecisionTreeClassifier(random_state=1234)
    model = dtc.fit(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    regressor = DecisionTreeRegressor()
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    df=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
    print(df)
    text_representation = tree.export_text(clf)
    print(text_representation)

    with open("tree.log", "w") as fout:
        fout.write(text_representation)
def choose_x_y():

    e = LabelEncoder()
    dataset = pd.read_csv("outputting/output2.csv")
    dataset['0'] = dataset['0'].astype('category')
    dataset['0'] = dataset['0'].cat.codes
    dataset['2'] = dataset['2'].astype('category')
    dataset['2'] = dataset['2'].cat.codes
    dataset['3'] = dataset['3'].astype('category')
    dataset['3'] = dataset['3'].cat.codes
    dataset['5'] = dataset['5'].astype('category')
    dataset['5'] = dataset['5'].cat.codes
    dataset['6'] = dataset['6'].astype('category')
    dataset['6'] = dataset['6'].cat.codes
    dataset['7'] = dataset['7'].astype('category')
    dataset['7'] = dataset['7'].cat.codes
    dataset['8'] = dataset['8'].astype('category')
    dataset['8'] = dataset['8'].cat.codes
    dataset['9'] = dataset['9'].astype('category')
    dataset['9'] = dataset['9'].cat.codes
    dataset['10'] = dataset['10'].astype('category')
    dataset['10'] = dataset['10'].cat.codes
    dataset['11'] = dataset['11'].astype('category')
    dataset['11'] = dataset['11'].cat.codes
    dataset.to_csv('ProcessedData.csv')

    x = dataset.drop(['4','1'], axis=1)
    y = dataset['4']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05, random_state=42)
    dtr1 = DecisionTreeRegressor()
    dtr1.fit(x_train, y_train)
    DecisionTreeRegressor(ccp_alpha=0.0, criterion='mse', max_depth=None,
                          max_features=None, max_leaf_nodes=None,
                          min_impurity_decrease=0.0, min_impurity_split=None,
                          min_samples_leaf=1, min_samples_split=2,
                          min_weight_fraction_leaf=0.0, presort='deprecated',
                          random_state=None, splitter='best')
# x_test is passed into the predict function to predict all values for the "y_test" and see if y_pred matches y_test.
# x_test takes a pd.array of values, 12 in total, each one representing a column in the "input1.csv" after the first
    y_pred = dtr1.predict(x_test)
    print(y_pred)
    plt.scatter(y_test, y_pred)
    plt.show()
if __name__ == '__main__':
    choose_x_y()