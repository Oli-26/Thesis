import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from keras.models import Sequential
from keras import layers
import numpy as np
from LoadData import load_from_file, load_new, split_by_project
from ModelTesting import split_data
from NLP import create_vectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
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

def train_lstm():
    number_of_examples = 50000
    df = load_from_file('technical_debt_dataset.csv', amount = number_of_examples)
    #df = load_new('file.csv', amount = number_of_examples)

    tokenizer = Tokenizer(num_words=None,filters = '!##$%&()*+', lower = True, split = ' ')
    tokenizer.fit_on_texts(df['commenttext'])
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    
    from keras.preprocessing.sequence import pad_sequences
    X = tokenizer.texts_to_sequences(df['commenttext'])
    X =  pad_sequences(X, 200)

    
    X_train, X_test, y_train, y_test = train_test_split(X, df['category_id'], random_state = 42, train_size = 0.75)
    y_train, y_test = categorize_output_vector(y_train, y_test)
    
    model = Sequential()
    #Embedding layer that expects input dimension of 201, outputs 50 features per word.
    model.add(layers.Embedding(len(word_index)+1, 50, input_length=X.shape[1]))
    #Dropout layer for sentence vectors
    model.add(layers.SpatialDropout1D(0.05))
    #Long short term memory layer
    model.add(layers.LSTM(25, dropout=0.05, recurrent_dropout=0.05))
    #Output layer for classificaiton
    model.add(layers.Dense(y_train.shape[1], activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer= 'adam',  metrics=['acc'])
    
    history = model.fit(X_train, y_train, epochs=30, batch_size=20,validation_data=(X_test, y_test), callbacks=[EarlyStopping(monitor='val_loss', patience=50, min_delta=0.0001)])
    plot(history)


    
def train_mlp():
    number_of_examples = 50000
    df = load_from_file('technical_debt_dataset.csv', amount = number_of_examples)
    Global_y = to_categorical(df['category_id'])
    
    listDf = split_by_project(df)
    
    
    
    #X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, train_size = 0.70)
    for dataframe in listDf:
        X = dataframe['commenttext']
        y = dataframe['category_id']
        
        
        tfidf = create_vectorizer(10)
        tfidf.fit(X)
        #input_len = len(tfidf.get_feature_names())
        #print(str(input_len) + " features found in " + str(len(X)) + " examples.")
    
        X_v = tfidf.transform(X)
        
        input_dim = X_v.shape[1]

        y = to_categorical(y)
        while y.shape[1] < Global_y.shape[1]:
            a = np.zeros((y.shape[0], 1))
            y = np.append(y, a, axis=1)
            
            
        model = Sequential()
        model.add(layers.Dense(25, activation='elu', input_dim = input_dim))
        model.add(layers.Dense(Global_y.shape[1], activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
        
        history = model.fit(X_v, y, epochs=10, verbose=False, batch_size=10)
        
        otherDataFrames = listDf

        
        prediction_store = []
        true_value_store = [] 
        
        for odf in otherDataFrames:
            if not pd.DataFrame.equals(odf, dataframe):
            
                test_x = odf['commenttext']
                test_y = odf['category_id']
                
                y_pred = model.predict(tfidf.transform(test_x))
                y_pred_bool = np.argmax(y_pred, axis=1)
                
                prediction_store = prediction_store + y_pred_bool.tolist()
                true_value_store = true_value_store + test_y.tolist()

        
        print(dataframe['project'].unique() + " (" + str(len(dataframe['project'])) + ")")

        print(classification_report(true_value_store, prediction_store, zero_division=0))
    #### Cross project validation
    
    
    
  
    
def train_cnn():
    number_of_examples = 50000
    df = load_from_file('technical_debt_dataset.csv', amount = number_of_examples)
    #df = load_new('file.csv', amount = number_of_examples)

    tokenizer = Tokenizer(num_words=None,filters = '!##$%&()*+', lower = True, split = ' ')
    tokenizer.fit_on_texts(df['commenttext'])
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    
    from keras.preprocessing.sequence import pad_sequences
    X = tokenizer.texts_to_sequences(df['commenttext'])
    X =  pad_sequences(X, 200)

    
    X_train, X_test, y_train, y_test = train_test_split(X, df['category_id'], random_state = 42, train_size = 0.75)
    y_train, y_test = categorize_output_vector(y_train, y_test)
    
    
    model = Sequential()
    #Embedding layer that expects input dimension of 201, outputs 50 features per word.
    model.add(layers.Embedding(len(word_index)+1, 50, input_length=X.shape[1]))
    #Dropout layer for sentence vectors
    model.add(layers.SpatialDropout1D(0.05))
    
    model.add(layers.Conv1D(60,5,activation='relu'))
    #Long short term memory layer
    model.add(layers.LSTM(25, dropout=0.05, recurrent_dropout=0.05))
    #Output layer for classificaiton
    model.add(layers.Dense(y_train.shape[1], activation='softmax'))
    
    
    model.compile(loss='categorical_crossentropy', optimizer= 'adam',  metrics=['acc'])
    
    history = model.fit(X_train, y_train, epochs=5, batch_size=10,validation_data=(X_test, y_test), callbacks=[EarlyStopping(monitor='val_loss', patience=50, min_delta=0.0001)])
    
    classification(model, X_test, y_test)
    plot(history)


    
    
def classification(model, X_test_v, y_test):
    y_pred = model.predict(X_test_v, batch_size=64, verbose=1)
    y_pred_bool = np.argmax(y_pred, axis=1)
    
    rounded_y_test = np.argmax(y_test, axis=1)
    #print(confusion_matrix(rounded_y_test, y_pred_bool))
    print(classification_report(rounded_y_test, y_pred_bool))
    
def plot(history):
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

#train_lstm()
train_mlp()
#train_cnn()    