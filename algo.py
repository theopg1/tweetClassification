import html

from joblib import dump

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
#from stop_words import get_stop_words
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

df = pd.read_csv("labels.csv")

tweets = df["tweet"].transform(lambda line: html.unescape(line))

df.drop('tweet', axis=1, inplace=True)

X = tweets
y = df[ ["class"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.70, random_state=15)

clf = make_pipeline(
    #TfidfVectorizer(stop_words=get_stop_words('en')),
    TfidfVectorizer(stop_words='english'),
    OneVsRestClassifier(SVC(kernel='linear', probability=True))
)

clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))

dump(clf,"test.joblib")