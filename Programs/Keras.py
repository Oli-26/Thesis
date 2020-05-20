import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from keras.models import Sequential
from keras import layers
import numpy as np
from LoadData import load_from_file, load_new
from ModelTesting import split_data
from NLP import create_vectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import EarlyStopping, ModelCheckpoint

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
    from keras.preprocessing.text import Tokenizer
    number_of_examples = 50000
    #df = load_from_file('technical_debt_dataset.csv', amount = number_of_examples)
    df = load_new('file.csv', amount = number_of_examples)

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
    
    # Define an optimizer
    #from keras.optimizers import SGD
    #opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    
    model.compile(loss='categorical_crossentropy', optimizer= 'adam',  metrics=['acc'])
    
    history = model.fit(X_train, y_train, epochs=30, batch_size=20,validation_data=(X_test, y_test), callbacks=[EarlyStopping(monitor='val_loss', patience=50, min_delta=0.0001)])
    #callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)]
    plot(history)


    
def train_mlp():
    number_of_examples = 50000

    #df = load_from_file('technical_debt_dataset.csv', amount = number_of_examples)
    
    df = load_new('file.csv', amount = number_of_examples)
    X = df['commenttext']
    y = df['category_id']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, train_size = 0.70)

    tfidf = create_vectorizer(10)
    tfidf.fit(X)
    input_len = len(tfidf.get_feature_names())
    print(str(input_len) + " features found")
    X_train_v = tfidf.transform(X_train)
    X_test_v = tfidf.transform(X_test)
    
    input_dim = X_train_v.shape[1]
    y_train, y_test = categorize_output_vector(y_train, y_test)

    model = Sequential()
    model.add(layers.Dense(25, activation='elu', input_dim = input_dim))
    #model.add(layers.Dropout(0.1))
    #model.add(layers.Dense(25, activation='elu'))
    model.add(layers.Dense(y_train.shape[1], activation='softmax'))

    from keras.optimizers import SGD
    #opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    callbacks = [EarlyStopping(monitor='val_loss', patience=10)]
    history = model.fit(X_train_v, y_train, epochs=1000, verbose=False, validation_data=(X_test_v, y_test), batch_size=25, callbacks = callbacks)

    loss, train_accuracy = model.evaluate(X_train_v, y_train, verbose=False)
    print("Training Accuracy: {:.4f}".format(train_accuracy))
    loss, test_accuracy = model.evaluate(X_test_v, y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}\n".format(test_accuracy))

    
    from sklearn.metrics import classification_report
    y_pred = model.predict(X_test_v, batch_size=64, verbose=1)
    #print(y_pred)
    y_pred_bool = np.argmax(y_pred, axis=1)
    print(y_pred_bool)
    #print(classification_report(y_test, y_pred_bool))


    plot(history)

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

train_lstm()
#train_mlp()    