
import pandas as pd
from io import StringIO
import re
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from NLP import create_vectorizer
    

def train_logistic_regression(X_train, y_train, min_df): 
    ## Create bag of words.
    tfidf = create_vectorizer(min_df)

    ## Pipe functions together to create pipeable model.
    from sklearn.pipeline import Pipeline
    model = Pipeline([('tfidf', tfidf), ('lg', LogisticRegression(random_state=0))])
    
    model.fit(X_train, y_train)
    return model
   
