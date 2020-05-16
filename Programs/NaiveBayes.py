import pandas as pd
from io import StringIO
import re

#Define file name 
address = 'technical_debt_dataset.csv'




def print_categories(df):
    # Conversion form classification to category id(Useful for deciphering output)
    category_id_df = df[['classification', 'category_id']].drop_duplicates().sort_values('category_id')
    category_to_id = dict(category_id_df.values)
    id_to_category = dict(category_id_df[['category_id', 'classification']].values)
    df.head()
    print(id_to_category)

def plot_classification_frequency(df):
    # Plot count of different types
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8,6))
    df.groupby('classification').commenttext.count().plot.bar(ylim=0)
    plt.show()


    
def extract_features(df):
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
    features = tfidf.fit_transform(df.commenttext).toarray()
    labels = df.category_id
    print(features.shape)

def train_model(df):
    #Create learning model
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.naive_bayes import MultinomialNB
    X_train, X_test, y_train, y_test = train_test_split(df['commenttext'], df['category_id'], random_state = 14)

    # Pipe countvectorizer, tfidftransformer, and multinomial naivebayes together.
    from sklearn.pipeline import Pipeline
    text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
    text_clf.fit(X_train, y_train)
    
    
    # return pipe.
    return text_clf


def examples(clf):
    # Test examples
    print(clf.predict(["This is poorly designed. We should change the char x to an int and then cast it later."]))


 
    
    
    
    
#MainBody
from LoadData import load_from_file

# Load dataframe
df = load_from_file(address, amount = 20000)

extract_features(df)
clf = train_model(df)
examples(clf)
