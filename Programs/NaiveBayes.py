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
        
        ## This shows us examples of debt type 1 (design)
        #comments_with_prediction = zip(X_test, general_predicted)
        #for x in comments_with_prediction:
        #    if(x[1] == 1):
        #        print(x)
        print("---------------------------\n\n")
        
    return (np.mean(train_predicted == y_train), np.mean(general_predicted == y_test))
    
def train_naive_bayes(X_train, y_train, min_words): 
    ## Create bag of words.
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=min_words, norm='l2', encoding='latin-1', ngram_range=(1, 2))

    ## Pipe functions together to create pipeable model.
    from sklearn.pipeline import Pipeline
    text_clf = Pipeline([('tfidf', tfidf), ('clf', MultinomialNB())])
    
    
    text_clf.fit(X_train, y_train)
    return text_clf
   
def naive_bayes_model(df, verbose, min_words):
    if verbose:
        extract_features(df)
        
    ## Split dataset into training and testing. 3/1 ratio split.    
    X_train, X_test, y_train, y_test = train_test_split(df['commenttext'], df['category_id'], random_state = 10, train_size = 0.25)
    
    text_clf = train_naive_bayes(X_train, y_train, min_words)
    acc = test_model(verbose, text_clf, X_train, X_test, y_train, y_test)
    return (text_clf, acc)
    
    
def examples(clf):
    # Test examples
    print(clf.predict(["This is poorly designed. We should change the char x to an int and then cast it later."]))
 
def main(): 
     
    ## MainBody
    from LoadData import load_from_file
    ## Init variables
    address = 'technical_debt_dataset.csv'
    number_of_examples = 70000
    verbose = True
    ## Load dataframe
    df = load_from_file(address, amount = number_of_examples)


    ##Lets take a look at out dataframe
    print(df)
    print("\n")   


    print_categories(df)
    number_of_examples = (df.shape[0])

    plot_classification_frequency(df, number_of_examples, verbose = verbose)
    clf = naive_bayes_model(df, verbose, min_words = 10)
    #examples(clf)
