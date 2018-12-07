from flask import Flask, flash, redirect, render_template, request, session, abort,url_for
import os
from sqlalchemy.orm import sessionmaker
from tabledef import *


#Machine Learning Code
## Import libraries
import numpy as np
import pandas as pd
import sqlite3    ## SQL Interface

from sklearn.feature_extraction.text import CountVectorizer  ## BOW Model

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
conn = sqlite3.connect('final.sqlite')  #Loading the sqlite file for future use
final = pd.read_sql_query("""SELECT * FROM Reviews""", conn)
conn.close()
final.drop(['index'],axis=1,inplace = True)

final.drop(['SEM 5 SGPA','SEM 5 KT','SEM 6 SGPA', 'SEM 6 KT', 'SEM 7 SGPA', 'SEM 7 KT', 'SEM 8 SGPA',"Teacher's Feedback","Cleaned_Feedback",'Average pointer'],axis=1,inplace = True)
X = final.iloc[:,:20].values
Y = pd.DataFrame(X)
Y['Social_Skills'] = final['Scocial_Skills']
Y['Label'] = final['Label']
Y.dropna(axis = 0, inplace = True)
X = Y.iloc[:,:21].values
y = Y['Label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, shuffle = False) #Splitting X and Y as 70 % training and 30 % testing with shuffle set to false
# Values for the hyperparameter 'C':
tuned_params = [{'C': [0.0001,0.001,0.01,0.1,1,10,100,1000,10000]}]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, shuffle = False)# Grid Search
model = GridSearchCV(LogisticRegression(), tuned_params, scoring = 'accuracy')
model.fit(X_train, y_train)
print(model.best_estimator_)
print(model.score(X_test, y_test))
clf = LogisticRegression(C = 0.01)
clf.fit(X_train, y_train)


#Flask CODE
engine = create_engine('sqlite:///tutorial.db', echo=True)
 
app = Flask(__name__)
 
@app.route('/')
def home():
    '''
    if not session.get('logged_in'):
        return render_template('login.html')
    else:
    '''
    return render_template('form.html')

@app.route('/login', methods=['POST'])
def do_admin_login():

    POST_USERNAME = str(request.form['username'])
    POST_PASSWORD = str(request.form['password'])

    Session = sessionmaker(bind=engine)
    s = Session()
    query = s.query(User).filter(User.username.in_([POST_USERNAME]), User.password.in_([POST_PASSWORD]) )
    result = query.first()
    if result:
        session['logged_in'] = True
    else:
        flash('wrong password!')
    return home()

@app.route('/submit', methods = ['GET','POST'])
def final():
    if request.method == 'POST':
        name = str(request.form['element_1'])
        gender = int(request.form['element_20'])
        sem1_ptr = float(request.form['element_2'])
        sem1_kt = int(request.form['element_11'])
        sem2_ptr = float(request.form['element_3'])
        sem2_kt = int(request.form['element_12'])
        sem3_ptr = float(request.form['element_4'])
        sem3_kt = int(request.form['element_13'])
        sem4_ptr = float(request.form['element_5'])
        sem4_kt = int(request.form['element_18'])
        assign = int(request.form['element_22'])
        travel = int(request.form['element_23'])
        studies = int(request.form['element_24'])
        attnd = int(request.form['element_25_1'])
        internet = int(request.form['element_26_1'])
        speed = int(request.form['element_27_1'])
        mode_trns = int(request.form['element_28_1'])
        lectures_2 = int(request.form['element_29_1'])
        submissions = int(request.form['element_30_1'])
        lectures_5 = int(request.form['element_31_1'])
        practicals_5 = int(request.form['element_32_1'])
        coaching_class = int(request.form['element_33_1'])
        social_skills = int(request.form['element_34_1'])
        X = np.array([sem1_ptr,sem1_kt,sem2_ptr,sem2_kt,sem3_ptr,sem3_kt,sem4_ptr,sem4_kt,assign,travel,studies,attnd,internet,speed,mode_trns,lectures_2,submissions,lectures_5,practicals_5,coaching_class,social_skills])
        y_pred = clf.predict([X])
        print(y_pred[0])
    return redirect((url_for('result', valr =str(y_pred[0]),sname= name)))

@app.route("/result/<valr>/<sname>")
def result(valr,sname):
    return render_template("submit.html",pred = valr,name= sname)


@app.route("/logout")
def logout():
    session['logged_in'] = False
    return home()
 
if __name__ == "__main__":
    app.secret_key = os.urandom(12)
    app.run(debug=True)
    