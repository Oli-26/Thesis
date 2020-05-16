
import pandas as pd
from io import StringIO
import re
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer

    
def train_svm(X_train, y_train, min_words): 
    ## Create bag of words.
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=min_words, norm='l2', encoding='latin-1', ngram_range=(1, 2), max_df=0.85)

    ## Pipe functions together to create pipeable model.
    
    from sklearn.neighbors import KNeighborsClassifier

    from sklearn.pipeline import Pipeline
    model = Pipeline([('tfidf', tfidf), ('svm', svm.SVC())])
    
    model.fit(X_train, y_train)
    return model
   
