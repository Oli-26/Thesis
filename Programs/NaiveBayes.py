import pandas as pd
from io import StringIO
import re
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

def print_categories(df):
    ## Conversion form classification to category id(Useful for deciphering output)
    category_id_df = df[['classification', 'category_id']].drop_duplicates().sort_values('category_id')
    category_to_id = dict(category_id_df.values)
    id_to_category = dict(category_id_df[['category_id', 'classification']].values)
    df.head()
    print(id_to_category)

def classification_frequency(df, number_of_examples):
    print("Statistics -")
    print(df.groupby('classification').commenttext.count()/(number_of_examples/100))
    print("-------------\n")
    
def extract_features(df):
    ## A way to see the shape of the bag of words. Does not effect the output. Duplicated code - Should find a nicer way to do this from pipeline output if possible
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2))
    features = tfidf.fit_transform(df.commenttext).toarray()
    labels = df.category_id
    print("Feature set shape: " + str(features.shape) + "\n")
    

    
def train_naive_bayes(X_train, y_train, min_words): 
    ## Create bag of words.
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=min_words, norm='l2', encoding='latin-1', ngram_range=(1, 2))

    ## Pipe functions together to create pipeable model.
    from sklearn.pipeline import Pipeline
    text_clf = Pipeline([('tfidf', tfidf), ('clf', MultinomialNB())])
    
    text_clf.fit(X_train, y_train)
    return text_clf
   
def naive_bayes_model(X_train, y_train, verbose, min_words):
    if verbose:
        extract_features(df)

    text_clf = train_naive_bayes(X_train, y_train, min_words)

    return text_clf
    
    
