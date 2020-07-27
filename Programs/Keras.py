import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from keras.models import Sequential
from keras import layers, Input, Model
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
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer 

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
    for comment in df['commenttext']:
        tokens = tokenizer.tokenize(comment)
        processed_comments.append(tokens)
    
    tokenizer = Tokenizer(num_words=None,filters = '!##$%&()*+', lower = True, split = ' ')
    tokenizer.fit_on_texts(processed_comments)
    X = tokenizer.texts_to_sequences(processed_comments)
    X = pad_sequences(X, 300)    

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    
    # Prepare embedding matrix
    words_not_found = []
    nb_words = len(word_index)+1
    embedding_matrix = np.zeros((nb_words, embed_dim))    
    for word, i in word_index.items():
        if i >= nb_words:
            continue
        embedding_vector = embedding_index.get(word)
        if(embedding_vector is not None) and len(embedding_vector) > 0:
            embedding_matrix[i] = embedding_vector
        else:
            words_not_found.append(word)
    print('number of null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
    return (X, embedding_matrix, nb_words)
    
def train_lstm():
    number_of_examples = 1000
    df = load_from_file('technical_debt_dataset.csv', amount = number_of_examples)
    nb_words = 5000
    embed_dim = 300
    
    X, embedding_matrix, nb_words = embed_words(df, nb_words = nb_words, embed_dim = embed_dim)
    X_train, X_test, y_train, y_test = train_test_split(X, df['category_id'], random_state = 42, train_size = 0.75)
    y_train, y_test = categorize_output_vector(y_train, y_test)
    
    model = Sequential()
    #Embedding layer
    model.add(layers.Embedding(nb_words, embed_dim, weights=[embedding_matrix], input_length=300))
    #Dropout layer for sentence vectors
    model.add(layers.SpatialDropout1D(0.05))
    #Long short term memory layer
    model.add(layers.LSTM(25, dropout=0.05, recurrent_dropout=0.05))
    #Output layer for classificaiton
    model.add(layers.Dense(y_train.shape[1], activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer= 'adam',  metrics=[get_f1])
    
    history = model.fit(X_train, y_train, epochs=10, batch_size=10,validation_data=(X_test, y_test), callbacks=[EarlyStopping(monitor='val_loss', patience=5, min_delta=0.0001)])

    classification(model, X_test, y_test)
    plot([history])
    
def train_mlp():
    df = load_from_file('technical_debt_dataset.csv', amount = 500)
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
        
        test_x = test['commenttext']
        test_y = test['category_id']
        
        # Create vectorizer for words. Use this to determine input shape of predictor network. 
        tfidf = create_vectorizer(10)
        tfidf.fit(X)
        X_v = tfidf.transform(X)
        input_dim = X_v.shape[1]
        
        # Convert integer values of y to lists where int is implicit by index.
        y = to_categorical(y)
        test_y = to_categorical(test_y)
        # If the training set is missing categorisation from the global prediction, then add row of 0's to include this category to predition.
        while y.shape[1] < Global_y.shape[1]:
            a = np.zeros((y.shape[0], 1))
            y = np.append(y, a, axis=1)
            
            
        while test_y.shape[1] < Global_y.shape[1]:
            a = np.zeros((test_y.shape[0], 1))
            test_y = np.append(test_y, a, axis=1)
            
            
        model = Sequential()
        dense1 = layers.Dense(25, activation='elu', input_dim = input_dim)
        model.add(dense1)
        model.add(layers.Dense(Global_y.shape[1], activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[get_f1])
        
        print("Begin model fitting")
        history = model.fit(X_v, y, epochs=10, verbose=False, batch_size=10, validation_data = (tfidf.transform(test_x), test_y))
        Histories.append(history)
        print("Model fitting complete")

        y_pred = model.predict(tfidf.transform(test_x))
        y_pred_bool = np.argmax(y_pred, axis=1)     

        print(classification_report(test['category_id'], y_pred_bool, zero_division=0))
        
        weights = dense1.get_weights()
        words = tfidf.get_feature_names()
        zipped = list(zip(words, weights))
        
        #sorted_list = sorted(zipped, key = lambda x: sum(list(x[1][0])))
        #for i in range(0, min(5,len(sorted_list))):
        #    print(str(sorted_list[i][0]))
    
    plot(Histories)
    
def train_cnn():
    number_of_examples = 10000
    df = load_new('file.csv', amount = number_of_examples, binary=True)

    embed_dim = 300
    max_words = 310
    filters = 300
    unigram_poolsize = 2
    bigram_poolsize = 2
    Global_y = to_categorical(df['category_id'])
    unique = np.unique(df['project'])
    Histories = []
    embedding_index = load_embeddings()
    
    for i in range(0, len(unique)):
        print("Running for test project " + str(unique[i]))
        newDF = df[df['project'] != unique[i]]
        test = df[df['project'] == unique[i]]
        
        X, embedding_matrix, nb_words = embed_words(df, embed_dim = embed_dim, embedding_index = embedding_index)  
        
        tokenizer = RegexpTokenizer(r'\w+')
        processed_comments_train = []
        processed_comments_test = []
        for comment in newDF['commenttext']:
            tokens = tokenizer.tokenize(comment)
            processed_comments_train.append(tokens)
        for comment in test['commenttext']:
            tokens = tokenizer.tokenize(comment)
            processed_comments_test.append(tokens)
        
        
        tokenizer = Tokenizer(num_words=None,filters = '!##$%&()*+', lower = True, split = ' ')
        tokenizer.fit_on_texts(processed_comments_train + processed_comments_test)
        X_train = tokenizer.texts_to_sequences(processed_comments_train)
        X_train = pad_sequences(X_train, max_words)    
        
        X_test = tokenizer.texts_to_sequences(processed_comments_test)
        X_test = pad_sequences(X_test, max_words)            
        
        y_train = newDF['category_id']
        y_test = test['category_id']
        
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, random_state = 42, train_size = 0.5)
        
        y_train, y_test = categorize_output_vector(y_train, y_test)

        input_shape = Input(shape=(max_words,))
        
        x = layers.Embedding(nb_words, embed_dim, weights = [embedding_matrix])(input_shape)
        x = layers.Dropout(rate = 0.01)(x)        
        
        y = (layers.Conv1D(filters,1,activation='relu'))(x)
        y = (layers.MaxPooling1D(pool_size=unigram_poolsize,strides=2, padding='valid'))(y)
        y = (layers.Flatten())(y)
        
        z = (layers.Conv1D(filters,2,activation='relu'))(x)
        z = (layers.MaxPooling1D(pool_size=bigram_poolsize,strides=2, padding='valid'))(z)
        z = (layers.Flatten())(z)
        
        x = layers.Concatenate(axis=1)([y, z])
        x = layers.Dense(1)(x)
        
        x = (layers.Dense(Global_y.shape[1], activation='softmax'))(x)
       
        model = Model(input_shape,x)
        print(model.summary())

        model.compile(loss='categorical_crossentropy', optimizer= 'adam',  metrics=[get_f1])
        history = model.fit(X_train, y_train, epochs=5, batch_size=10, validation_data = (X_val, to_categorical(y_val)))

        classification(model, X_test, y_test)
        Histories.append(history)
    plot(Histories)
    
def classification(model, X_test_v, y_test):
    y_pred = model.predict(X_test_v, batch_size=64, verbose=1)
    y_pred_bool = np.argmax(y_pred, axis=1)
    rounded_y_test = np.argmax(y_test, axis=1)
    print(classification_report(rounded_y_test, y_pred_bool))
    
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

#train_lstm()
#train_mlp()
train_cnn()    