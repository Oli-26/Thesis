import pandas as pd
from io import StringIO
import re
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

    

    
def train_knn(X_train, y_train, amount_neighbors, min_words): 
    ## Create bag of words.
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=min_words, norm='l2', encoding='latin-1', ngram_range=(1, 2))

    ## Pipe functions together to create pipeable model.
    


    from sklearn.pipeline import Pipeline
    text_clf = Pipeline([('tfidf', tfidf), ('knn', KNeighborsClassifier(n_neighbors=amount_neighbors))])
    
    text_clf.fit(X_train, y_train)
    return text_clf
   
def knn_model(X_train, y_train, verbose, min_words, amount_neighbors):
    
    ## Split dataset into training and testing. 3/1 ratio split.    
    X_train, X_test, y_train, y_test = train_test_split(df['commenttext'], df['category_id'], random_state = 10, train_size = 0.25)
    text_clf = train_knn(X_train, y_train, amount_neighbors = amount_neighbors, min_words = min_words)
    #acc = test_model(verbose, text_clf, X_train, X_test, y_train, y_test)
    return text_clf