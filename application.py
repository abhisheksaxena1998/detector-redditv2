import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import re
import flask
from flask import Flask, render_template, request

app=Flask(__name__)


REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,_;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')

flairs={1:"AskIndia",
2:"Non-Political",
3:"[R]eddiquette",
4:"Scheduled",
5:"Photography",
6:"Science/Technology",
7:"Politics",
8:"Business/Finance",
9:"Policy/Economy",
10:"Sports",
11:"Food",
12:"AMA"
}

@app.route('/')

@app.route('/index')
def index():
    return flask.render_template('index.html')

@app.route('/statistics')
def statistics():
    return flask.render_template('statistics.html')

@app.route("/register", methods=["POST"])
def register():
    if request.method=='POST':
        nm = request.form.get("url")
        mm=nm


        # Read in the data
        #df = pd.read_csv('test_classifier_Input_for_classifier_stable.csv')
        df = pd.read_csv('https://raw.githubusercontent.com/abhisheksaxena1998/web_app_ml/master/reddit_input_url.csv')
        #df.columns=['flair','title','score','id','url','comms_num','body','author','comments','timestamp']  
        df.dropna(inplace=True)
        df.columns=['index','url','flair'] 
        # Sample the data to speed up computation
        # Comment out this line to match with lecture
        #df = df.sample(frac=0.1, random_state=10)

        #df.head()
        print (df)
        X_train, X_test, y_train, y_test = train_test_split(df['url'],df['flair'],random_state=0)
        vect = CountVectorizer().fit(X_train)
        print (vect.get_feature_names())

        X_train_vectorized = vect.transform(X_train)

        X_train_vectorized

        print ((X_train_vectorized.shape))

        # Train the model
        model = LogisticRegression(solver='lbfgs', multi_class='auto')
        model.fit(X_train_vectorized, y_train)

        #print(model.predict(vect.transform(['https  wwwredditcom r india comments bemcxg clueless american 11 etiquette with our indian '])))
        filename='url_model.pkl'

        pickle.dump(model, open(filename, 'wb'))
        load_lr_model =pickle.load(open(filename, 'rb'))
        #text="https://www.reddit.com/r/india/comments/1s57oi/need_feedback_for_insurance_policy_that_i_took/"
        #text = REPLACE_BY_SPACE_RE.sub(' ', text)
        #text = BAD_SYMBOLS_RE.sub('', text)
        #print (text)
        #print (load_lr_model.predict(vect.transform([text])))

        nm=REPLACE_BY_SPACE_RE.sub(' ', nm)
        nm = BAD_SYMBOLS_RE.sub('', nm)
        pred=load_lr_model.predict(vect.transform([nm]))
    return flask.render_template('result.html',prediction=flairs[int(pred)],url=mm)
