import pandas as pd
from io import StringIO
import re
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from NLP import create_vectorizer
    

    
def train_knn(X_train, y_train, amount_neighbors, min_df): 
    ## Create bag of words.
    tfidf = create_vectorizer(min_df)

    ## Pipe functions together to create pipeable model.
    from sklearn.pipeline import Pipeline
    model = Pipeline([('tfidf', tfidf), ('knn', KNeighborsClassifier(n_neighbors=amount_neighbors))])
    
    model.fit(X_train, y_train)
    return model
   
