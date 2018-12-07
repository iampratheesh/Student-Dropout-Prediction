## Import libraries
import numpy as np
import pandas as pd

import sqlite3    ## SQL Interface

from sklearn.feature_extraction.text import CountVectorizer  ## BOW Model

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

conn = sqlite3.connect('final.sqlite')  #Loading the sqlite file for future use
final = pd.read_sql_query("""SELECT * FROM Reviews""", conn)
conn.close()
final.drop(['index'],axis=1,inplace = True)

bow_vect = CountVectorizer()
bow = bow_vect.fit_transform(final["Cleaned_Feedback"].values)

bow_vect = CountVectorizer()
bow = bow_vect.fit_transform(final["Cleaned_Feedback"].values)

X = final.iloc[:,:27].values

a = bow.toarray()
X = np.append(X,a, axis = 1)

Y = pd.DataFrame(X)
Y['Label'] = final['Label']
Y.dropna(axis = 0, inplace = True)

X = Y.iloc[:,:67].values
y = Y['Label'].values

tuned_params = [{'C': [0.0001,0.001,0.01,0.1,1,10,100,1000,10000]}]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, shuffle = False)

# Grid Search
model = GridSearchCV(LogisticRegression(), tuned_params, scoring = 'accuracy')
model.fit(X_train, y_train)

clf = LogisticRegression(C = 0.0001)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)*float(100)
print(acc)


