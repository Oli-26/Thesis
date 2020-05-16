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
    model = Pipeline([('tfidf', tfidf), ('knn', KNeighborsClassifier(n_neighbors=amount_neighbors))])
    
    model.fit(X_train, y_train)
    return model
   
def knn_model(X_train, y_train, verbose, min_words, amount_neighbors):
    model = train_knn(X_train, y_train, amount_neighbors = amount_neighbors, min_words = min_words)
    return model