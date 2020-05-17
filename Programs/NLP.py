
from sklearn.feature_extraction.text import TfidfVectorizer

def create_vectorizer(min_df):
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=min_df, norm='l2', encoding='latin-1', ngram_range=(1, 2), max_df=0.85)
    
    return tfidf