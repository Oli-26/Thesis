import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import keras
import keras.backend as K
from nltk.tokenize import RegexpTokenizer 
from keras.preprocessing.text import Tokenizer
from keras import layers, Input, Model
from keras.preprocessing.sequence import pad_sequences
from LoadData import load_from_file, load_new
from Keras import get_f1
import keras.backend as K
import numpy as np
from nltk.corpus import stopwords
from scipy.special import softmax
import tensorflow as tf
import math
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

    
def tokenize_input(df):    
    # We feed a comment into a trained CNN model to obtain the feature vector X in R(mX1) of this comment in the fully connected layer. Each feature x_i in the feature vector X corresponds to a filter.
    tokenizer = RegexpTokenizer(r'\w+')
    processed_comments = []
    cachedStopWords = stopwords.words("english")
    for comment in df['commenttext']:
        tokens = tokenizer.tokenize(comment)
        text = ' '.join([word for word in tokens if word not in cachedStopWords])
        processed_comments.append(text)
    tokenizer = Tokenizer(num_words=None,filters = '!##$%&()*+', lower = True, split = ' ')
    tokenizer.fit_on_texts(processed_comments)
    X = tokenizer.texts_to_sequences(processed_comments)
    X = pad_sequences(X, 1000)
    # Create reverse word map
    word_index = tokenizer.word_index
    reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
    return X, reverse_word_map
    


def extract_key_words(df):
    # Load model 
    #model = keras.models.load_model('newest', custom_objects={'get_f1':get_f1})
    model = keras.models.load_model('olddata', custom_objects={'get_f1':get_f1})
    model.summary()

    X, reverse_word_map = tokenize_input(df) 
    
    # Create map between input and hidden layers
    cnn_layer_1 = Model(model.input, model.layers[len(model.layers)-8].output) # uni
    cnn_layer_2 = Model(model.input, model.layers[len(model.layers)-7].output) # bi
    feature_vector_output = Model(model.input, model.layers[len(model.layers)-2].output)
    dense_layer = Model(model.input, model.layers[len(model.layers)-1].output)
    
    word_dict = {}
    X = X[:10000]

    weights = [layer.get_weights() for layer in model.layers]
    output_weights = weights[len(weights)-1][0]

    for offset in range(0, len(X)):
        X2 = X[offset:(offset+1)]
        dense = dense_layer.predict(X2)
        
        if np.argmax(dense) == 1:
            cnn_out_1 = cnn_layer_1.predict(X2)
            cnn_out_2 = cnn_layer_2.predict(X2)
            feature_vector = feature_vector_output.predict(X2)
            
            temp = [] 
            for i, f in enumerate(feature_vector[0]):
                w = f * output_weights[i]
                prob = softmax(w)
                temp.append((i, prob[1],w[1] ))
                
                
            for (ind, prob, w) in temp:   
                if prob > 0.5:
                    if(ind < 300):
                        #CNN1
                        for i in range(0, 1000):
                            if cnn_out_1[0,i, 0, ind] == feature_vector[0][ind]:
                                try:
                                    word = reverse_word_map[X2[0][i]]
                                    if word in word_dict.keys():
                                        word_dict[word].append(prob)
                                    else:
                                        word_dict[word] = [prob]
                                except:
                                    #print("Failed to find word")
                                    break;
                                break;
                    else:    
                        #CNN2
                        for i in range(0, 1000):
                            if cnn_out_2[0, i, 0, ind-300] == feature_vector[0][ind]:
                                try:
                                    word = reverse_word_map[X2[0][i]] + " " + reverse_word_map[X2[0][i+1]]
                                    if word in word_dict.keys():
                                        word_dict[word].append(prob)
                                    else:
                                        word_dict[word] = [prob] 
                                except:
                                    #print("Failed to find word")
                                    break;
                                break;
    
    output_array = []
    for key, a in word_dict.items():
        average = sum(a)/len(a)
        #print(key + ": " + str(average))
        output_array.append((key, average))
        
    output_array = sorted(output_array, key=lambda x: (-x[1]))
    
    for i in range(0, 50):
        print(output_array[i][0] + ": " + str(output_array[i][1]))
    
            
       

           
                 
    
    
    
type = "general"    
extract_key_words(load_new('file.csv', amount = 10000, type = type))
extract_key_words(load_from_file('technical_debt_dataset.csv', amount = 100000))

