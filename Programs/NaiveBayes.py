import pandas as pd
from io import StringIO
import re
import numpy as np
def print_categories(df):
    ## Conversion form classification to category id(Useful for deciphering output)
    category_id_df = df[['classification', 'category_id']].drop_duplicates().sort_values('category_id')
    category_to_id = dict(category_id_df.values)
    id_to_category = dict(category_id_df[['category_id', 'classification']].values)
    df.head()
    print(id_to_category)

def plot_classification_frequency(df, number_of_examples, verbose):
    ## Plot count of different types
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8,6))
    
    ## Print stats of dataset
    if(verbose):
        print("Statistics -")
        print(df.groupby('classification').commenttext.count()/(number_of_examples/100))
        print("-------------\n")
    
    
    
    df.groupby('classification').commenttext.count().plot.bar(ylim=0)
    plt.show()
    
def extract_features(df):
    ## A way to see the shape of the bag of words. Does not effect the output. Duplicated code - Should find a nicer way to do this from pipeline output if possible
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2))
    features = tfidf.fit_transform(df.commenttext).toarray()
    labels = df.category_id
    print("Feature set shape: " + str(features.shape) + "\n")
    
def train_model(df, verbose):
    if(verbose):
        extract_features(df)
  
    ##Create learning model
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.feature_extraction.text import TfidfVectorizer
        
    ## Split dataset into training and testing. 3/1 ratio split.    
    X_train, X_test, y_train, y_test = train_test_split(df['commenttext'], df['category_id'], random_state = 10, train_size = 0.25)
    
    ## Create bag of words.
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=12, norm='l2', encoding='latin-1', ngram_range=(1, 2))

    ## Pipe functions together to create pipeable model.
    from sklearn.pipeline import Pipeline
    text_clf = Pipeline([('tfidf', tfidf), ('clf', MultinomialNB())])
    text_clf.fit(X_train, y_train)
    
    ##Lets take a look at out dataframe
    print(df)
    print("\n")   
    
    print("Testing -----------------------")
    ##Lets predit to see the train accuracy of our model
    train_predicted = text_clf.predict(X_train)
    print("Train accuracy = " + str(np.mean(train_predicted == y_train)))
    
    ##Lets predict to see the generality of our model 
    general_predicted = text_clf.predict(X_test)    
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
    
    # return pipe.
    return text_clf
    
def examples(clf):
    # Test examples
    print(clf.predict(["This is poorly designed. We should change the char x to an int and then cast it later."]))
 
## MainBody
from LoadData import load_from_file
## Init variables
address = 'technical_debt_dataset.csv'
number_of_examples = 70000
verbose = True
## Load dataframe
df = load_from_file(address, amount = number_of_examples)

print_categories(df)
number_of_examples = (df.shape[0])

plot_classification_frequency(df, number_of_examples, verbose)
clf = train_model(df, verbose)
#examples(clf)
