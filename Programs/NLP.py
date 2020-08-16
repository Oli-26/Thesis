
from sklearn.feature_extraction.text import TfidfVectorizer

def create_vectorizer(min_df):
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=min_df, norm='l2', encoding='latin-1', ngram_range=(1, 2), max_df=0.85)
    return tfidf
    
    
from nltk.corpus import stopwords
from tqdm import tqdm
import codecs
import numpy as np
def load_embeddings():
    print('loading word embeddings...')
    embeddings_index = {}
    f = codecs.open('corpus.vec', encoding='utf-8')
    i = 0 
    for line in tqdm(f):
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
        #i = i + 1
        #if(i > 1000):
        #    break 
    f.close()
    return embeddings_index