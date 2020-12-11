#!/usr/bin/env python
# coding: utf-8

# In[19]:


from sklearn.preprocessing import StandardScaler
data = pd.read_csv('C:/Users/Justi/Project.tsv', error_bad_lines=False, sep='\t')
data = data.dropna()
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder 
from sklearn.compose import ColumnTransformer


le = preprocessing.LabelEncoder()
# 2/3. FIT AND TRANSFORM
# use df.apply() to apply le.fit_transform to all columns
data_2 = data.apply(le.fit_transform)
data_2.head()
enc = preprocessing.OneHotEncoder()
enc.fit(data_2)

onehotlabels = enc.transform(data_2).toarray()
onehotlabels.shape

print(list(data_2.columns))
print(data.dtypes)


feature_cols = ['num_ratings', 'num_reviews']
X = data_2[feature_cols] # Features
y = data_2.rating # Target variable


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.5,random_state=0)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(solver='liblinear')

# fit the model with data
logreg.fit(X_train,y_train)


y_pred=logreg.predict(X_test)
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)

logreg.score(X_train, y_train)


# In[18]:


class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

from sklearn.metrics import classification_report
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))



# In[17]:


from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt  # To visualize
import pandas as pd  # To read data
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import OneHotEncoder 
from sklearn.model_selection import train_test_split

data = pd.read_csv('C:/Users/Justi/Project.tsv', error_bad_lines=False, sep='\t') 
le = preprocessing.LabelEncoder()
data2 = data.apply(le.fit_transform)
enc = preprocessing.OneHotEncoder()
enc.fit(data2)

onehotlabels = enc.transform(data2).toarray()
onehotlabels.shape

data2.drop(['album_id', 'artist', 'name', 'date', 'genres','position'],axis=1,inplace=True)
print(list(data2.columns))
X = data2.iloc[:, 0].values.reshape(-1, 1)  # values converts it into a numpy array
Y = data2.iloc[:, 2].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.4,random_state=0)
linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(X, Y)  # perform linear regression
Y_pred = linear_regressor.predict(X)  # make predictions


plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()

r_sq = linear_regressor.score(X, Y)
print('coefficient of determination:', r_sq)
print('intercept:', linear_regressor.intercept_)
print('slope:', linear_regressor.coef_)



# In[17]:


import pandas as pd
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt
from sklearn import preprocessing

data = pd.read_csv('C:/Users/Justi/Project.tsv', error_bad_lines=False, sep='\t') 
le = preprocessing.LabelEncoder()
data2 = data.apply(le.fit_transform)
enc = preprocessing.OneHotEncoder()
enc.fit(data2)

onehotlabels = enc.transform(data2).toarray()
onehotlabels.shape


data2.drop(['album_id', 'artist', 'name', 'date', 'genres'],axis=1,inplace=True)

from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
chi_square_value,p_value=calculate_bartlett_sphericity(data2)
chi_square_value, p_value

from factor_analyzer.factor_analyzer import calculate_kmo
kmo_all,kmo_model=calculate_kmo(data2)

kmo_model

fa = FactorAnalyzer(rotation=None)
fa.fit(data2, 25)
ev, v = fa.get_eigenvalues()
ev

fa = FactorAnalyzer(rotation="varimax")
fa.fit(data2, 6)
fa.get_factor_variance()


# In[ ]:




