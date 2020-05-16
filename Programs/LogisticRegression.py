
import pandas as pd
from io import StringIO
import re
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

    

def train_logistic_regression(X_train, y_train, min_words): 
    ## Create bag of words.
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=min_words, norm='l2', encoding='latin-1', ngram_range=(1, 2))

    ## Pipe functions together to create pipeable model.
    from sklearn.pipeline import Pipeline
    text_clf = Pipeline([('tfidf', tfidf), ('lg', LogisticRegression(random_state=0))])
    
    text_clf.fit(X_train, y_train)
    return text_clf
   
def logistic_regression_model(X_train, y_train, verbose, min_words):

    text_clf = train_logistic_regression(X_train, y_train, min_words = min_words)

    return test_clf