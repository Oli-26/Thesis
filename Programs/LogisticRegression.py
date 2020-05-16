
import pandas as pd
from io import StringIO
import re
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

    
def test_model(verbose, text_clf, X_train, X_test, y_train, y_test):
    ##Lets predit to see the train accuracy of our model
    train_predicted = text_clf.predict(X_train)
    
    ##Lets predict to see the generality of our model 
    general_predicted = text_clf.predict(X_test)    

    if verbose:
        print("Testing -----------------------")
        print("Train accuracy = " + str(np.mean(train_predicted == y_train)))
        print("Test accuracy = " + str(np.mean(general_predicted == y_test)))
        #Show percentage of general prediction types 
        unique, counts = np.unique(general_predicted, return_counts=True)
        print(dict(zip(unique, counts*100/(len(general_predicted)))))
        print("---------------------------\n\n")
        
    return (np.mean(train_predicted == y_train), np.mean(general_predicted == y_test))
    
def train_logistic_regression(X_train, y_train, min_words): 
    ## Create bag of words.
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=min_words, norm='l2', encoding='latin-1', ngram_range=(1, 2))

    ## Pipe functions together to create pipeable model.
    from sklearn.pipeline import Pipeline
    text_clf = Pipeline([('tfidf', tfidf), ('lg', LogisticRegression(random_state=0))])
    
    text_clf.fit(X_train, y_train)
    return text_clf
   
def logistic_regression_model(df, verbose, min_words):
    ## Split dataset into training and testing. 3/1 ratio split.    
    X_train, X_test, y_train, y_test = train_test_split(df['commenttext'], df['category_id'], random_state = 10, train_size = 0.25)
    text_clf = train_logistic_regression(X_train, y_train, min_words = min_words)
    acc = test_model(verbose, text_clf, X_train, X_test, y_train, y_test)
    return (text_clf, acc)