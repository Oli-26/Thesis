import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from keras.models import Sequential
from keras import layers, Input, Model
from keras.callbacks import EarlyStopping
import numpy as np
from LoadData import load_from_file, load_new, split_by_project
from ModelTesting import split_data
from NLP import create_vectorizer, load_embeddings
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, confusion_matrix
from keras.utils.vis_utils import plot_model
import pandas as pd
import keras.backend as K
import nltk
#nltk.download()
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer 

import tensorflow as tf



def get_f1(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

def categorize_output_vector(y_train, y_test):
    #OneHotCode of outputs (used for multiclass classfication)
    from keras.utils import to_categorical
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # Quick fix for mismatched dimensions, adds an arrays of 0's for each absent category
    while y_train.shape[1] > y_test.shape[1]:
        a = np.zeros((y_test.shape[0], 1))
        y_test = np.append(y_test, a, axis=1)
    return y_train, y_test

def embed_words(df, embed_dim, embedding_index):
    tokenizer = RegexpTokenizer(r'\w+')
    processed_comments = []
    cachedStopWords = stopwords.words("english")
    for comment in df['commenttext']:
        tokens = tokenizer.tokenize(comment)
        text = ' '.join([word for word in tokens if word not in cachedStopWords])
        #text = tokens 
        processed_comments.append(text)
    
    tokenizer = Tokenizer(num_words=None,filters = '!##$%&()*+', lower = True, split = ' ')
    tokenizer.fit_on_texts(processed_comments)
    word_index = tokenizer.word_index

    # Prepare embedding matrix
    words_not_found = []
    nb_words = len(word_index)+1
    embedding_matrix = np.zeros((nb_words, embed_dim))    
    for word, i in word_index.items():
        if i >= nb_words:
            continue
        embedding_vector = embedding_index.get(word)
        if(embedding_vector is not None) and len(embedding_vector) > 0:
            embedding_matrix[i] = embedding_vector[:embed_dim]
        else:
            words_not_found.append(word)
    #print('number of null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
    return (embedding_matrix, nb_words, tokenizer)


    
def train_mlp():
    df = load_from_file('technical_debt_dataset.csv', amount = 100000)
    #df = load_new('file.csv', amount = 10000, type = "general")
    Global_y = to_categorical(df['category_id'])
    unique = np.unique(df['project'])
    Histories = []
    
    for i in range(0, len(unique)):
        print("Running for test project " + str(unique[i]))
        newDF = df[df['project'] != unique[i]]
        test = df[df['project'] == unique[i]]
        print("Train data = " + str(len(newDF)) + "  |  Test data = " + str(len(test)))
    
        X = newDF['commenttext']
        y = newDF['category_id']
        
        X_test = test['commenttext']
        y_test = test['category_id']
        
        # Create vectorizer for words. Use this to determine input shape of predictor network. 
        tfidf = create_vectorizer(10)
        tfidf.fit(X)
        X_v = tfidf.transform(X).toarray()
        X_test = tfidf.transform(X_test).toarray()
        input_dim = X_v.shape[1]

        # Convert integer values of y to lists where int is implicit by index.
        y = to_categorical(y)
        y_test = to_categorical(y_test)
     
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, random_state = 42, train_size = 0.5)    
            
        model = Sequential()
        model.add(layers.Dense(25, activation='elu', input_dim = input_dim))
        model.add(layers.Dense(Global_y.shape[1], activation='sigmoid'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[get_f1])
        
        print("Begin model fitting")
        history = model.fit(X_v, y, epochs=15, verbose=False, batch_size=35, validation_data = (X_val, y_val))
        Histories.append(history)
        print("Model fitting complete")
 

        #print(classification_report(y_test, y_pred_bool, zero_division=0))
        report = classification(model, X_test, y_test)
        print(report)

def types():
    types = ["general"]  #, "arch", "code", "build", "defect", "design", "documentation", "requirements", "test"]
    average = 0
    embedding_index = load_embeddings()
    for i in range(0, 10):
        #print("Running for type = " + types[i])
        average = average + cross_project_validation("general", embedding_index)
    average = average/10
    print("-------------------------------------------")
    print(str(average))
    print("-------------------------------------------")
           
def cross_project_validation(type, embedding_index):
    ## Trains and test CNN multiple times, leaving one project out for testing each time.
    df = load_new('file.csv', amount = 10000, type = type)
    print(df)
    #df = load_from_file('technical_debt_dataset.csv', amount = 100000, type = type)
    
    unique = np.unique(df['project'])

    Histories = []
    Models = []
    debt_f1 = []
    num_f1 = []
    for i in range(0, len(unique)):
        newDF = df[df['project'] != unique[i]]
        test = df[df['project'] == unique[i]]
        print("Running for test project " + str(unique[i]) + "(" + str(len(test['category_id'])) + "," + str(len(newDF['category_id'])) + ")")
        history, model, report = train_cnn(df, newDF, test, embedding_index)
        Histories.append(history)
        Models.append(model)
        debt_f1.append(report['1']['f1-score'])
        num_f1.append(report['1']['support'])
        
    av_debt = 0
    #for i in range(0, len(debt_f1)):
    #    av_debt = av_debt + debt_f1[i]*num_f1[i]
    #av_debt = av_debt / sum(num_f1)    
    av_debt = debt_f1[0]
    index = debt_f1.index(max(debt_f1))
    print("Average debt f1 score = " + str(av_debt))
    #plot(Histories)
    Models[index].save('m1')
    return av_debt

    
def train_cnn(df, newDF, test, embedding_index):  
    embed_dim = 300
    max_words = 1000
    filters = 300
    unigram_poolsize = max_words
    bigram_poolsize = max_words-1

    Global_y = to_categorical(df['category_id'])
    
    # Load embedding matrix
    embedding_matrix, nb_words, loaded_tokenizer = embed_words(df, embed_dim = embed_dim, embedding_index = embedding_index)  
    
    # Tokenize train and test comments.
    tokenizer = RegexpTokenizer(r'\w+')
    processed_comments_train = []
    processed_comments_test = []
    
    cachedStopWords = stopwords.words("english")
    for comment in newDF['commenttext']:
        tokens = tokenizer.tokenize(comment)
        text = ' '.join([word for word in tokens if word not in cachedStopWords])
        #text = tokens
        processed_comments_train.append(text)
    for comment in test['commenttext']:
        tokens = tokenizer.tokenize(comment)
        text = ' '.join([word for word in tokens if word not in cachedStopWords])
        #text = tokens
        processed_comments_test.append(text)
        
    X_train = loaded_tokenizer.texts_to_sequences(processed_comments_train)
    X_train = pad_sequences(X_train, max_words)    
    X_test = loaded_tokenizer.texts_to_sequences(processed_comments_test)
    X_test = pad_sequences(X_test, max_words)            
  
    y_train = newDF['category_id']
    y_test = test['category_id']
 
    # Convert output binary values into length 2 array (one hot vector incoding)
    y_train, y_test = categorize_output_vector(y_train, y_test)
    
    # Split into training validation and final (test) validation sets.
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, random_state = 42, train_size = 0.5)

    # Create network
    input_shape = Input(shape=(max_words,))
    
    x = layers.Embedding(nb_words, embed_dim, weights = [embedding_matrix])(input_shape)
    x = layers.Reshape((max_words, embed_dim, 1))(x)
    #x = layers.Dropout(rate = 0.001)(x)        
    
    ## Branch 1 (Unigram)
    y = (layers.Conv2D(filters,kernel_size=(1, embed_dim),activation='relu'))(x)
    y = (layers.MaxPooling2D(pool_size=(unigram_poolsize,1),strides=(1,1), padding='valid'))(y)
    y = (layers.Flatten())(y)
    
    ## Branch 2 (Bigram)
    z = (layers.Conv2D(filters,kernel_size=(2, embed_dim),activation='relu'))(x)
    z = (layers.MaxPooling2D(pool_size=(bigram_poolsize,1),strides=(1,1), padding='valid'))(z)
    z = (layers.Flatten())(z)
    
    ## Gather branches
    x = layers.Concatenate(axis=1)([y, z])
    x = layers.Dense(1)(x)
    x = (layers.Dense(Global_y.shape[1], activation='sigmoid'))(x)
   
    model = Model(input_shape,x)  
    callback = EarlyStopping(patience=2)
    model.compile(loss='binary_crossentropy', optimizer= 'adam',  metrics=[get_f1])
    history = model.fit(X_train, y_train, epochs=15, batch_size=36, verbose = False ,callbacks=[callback], validation_data = (X_val, y_val))
    
    print(model.summary())
    report = classification(model, X_test, y_test)
    
    #print(report)
    return (history, model, report)
    
def classification(model, X_test_v, y_test):
    y_pred = model.predict(X_test_v, batch_size=64, verbose=1)
    y_pred_bool = np.argmax(y_pred, axis=1)
    rounded_y_test = np.argmax(y_test, axis=1)
    report = classification_report(rounded_y_test, y_pred_bool, output_dict=True)
    print(classification_report(rounded_y_test, y_pred_bool))
    return report
    
def plot(historys):
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
    plt.figure(figsize=(12, 5))
    i = 1
    for history in historys:
        f1 = history.history['get_f1']
        val_f1 = history.history['val_get_f1']
        x = range(1, len(f1) + 1)
        plt.plot(x, f1, 'b', label='F1 - partition ' + str(i))
        plt.plot(x, val_f1, 'r', label='Val_F1 - partition' + str(i))
        plt.title('F1 score')
        plt.legend()
        i = i +1
    plt.show()


#train_mlp()
#train_cnn()  
#types()  