
import pandas as pd
from io import StringIO
import re
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.feature_extraction.text import TfidfVectorizer

    
   
def train_decision_tree(X_train, y_train, min_words): 
    ## Create bag of words.
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=min_words, norm='l2', encoding='latin-1', ngram_range=(1, 2))

    ## Pipe functions together to create pipeable model.
    from sklearn.pipeline import Pipeline
    model = Pipeline([('tfidf', tfidf), ('dtc', DecisionTreeClassifier())])
    
    model.fit(X_train, y_train)
    return model
   
def decision_tree_model(X_train, y_train, verbose, min_words):
    model = train_decision_tree(X_train, y_train, min_words = min_words)
    return model