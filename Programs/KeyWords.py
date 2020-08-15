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
def append_to_list(list_1, item):
    # List of the form (word, sum prob, sum weight, number of instances)
    # Item of the form (word, prob, weight)
    for x in range(0, len(list_1)):
        if list_1[x][0] == item[0]:
            list_1[x] = (list_1[x][0], list_1[x][1] + item[1], list_1[x][2] + item[2], list_1[x][3] + 1)
            return list_1
        
    list_1.append((item[0], item[1], item[2], 1))
    return list_1

    
    
    
def list_print(input_list):
    for x in range(0,len(input_list)):
        print("[" + str(x) + "] " + str(input_list[x][0]) + "(" + str(input_list[x][3]) + ") = " + str(input_list[x][1]*100) + "%")
    

    
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
    


def extract_key_words(df, type):
    print("extracting key words for " + type)
    K.clear_session()
    
    # Load model 
    model = keras.models.load_model('m2', custom_objects={'get_f1':get_f1})
    model.summary()
    # We feed a comment into a trained CNN model to obtain the feature vector X in R(mX1) of this comment in the fully connected layer. Each feature x_i in the feature vector X corresponds to a filter.
    tokenizer = RegexpTokenizer(r'\w+')
    
    # tokenize dataset
    cachedStopWords = stopwords.words("english")
    
    processed_comments = []
    for comment in df['commenttext']:
        tokens = tokenizer.tokenize(comment)
        text = ' '.join([word for word in tokens if word not in cachedStopWords])
        processed_comments.append(text)
        
    tokenizer = Tokenizer(num_words=None,filters = '!##$%&()*+', lower = True, split = ' ')
    tokenizer.fit_on_texts(processed_comments)
    X = tokenizer.texts_to_sequences(processed_comments)
    X = pad_sequences(X, 1000)
    Y = df['category_id']
    
    
    # Create reverse word map
    word_index = tokenizer.word_index
    reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
    
    # Create map between input and hidden layers
    cnn_layer = Model(model.input, model.layers[len(model.layers)-9].output) # uni
    max_pooling_layer = Model(model.input, model.layers[len(model.layers)-7].output) # uni
    
    cnn_layer_2 = Model(model.input, model.layers[len(model.layers)-8].output) # bi
    max_pooling_layer_2 = Model(model.input, model.layers[len(model.layers)-6].output) # bi
    
    feature_vector_output = Model(model.input, model.layers[len(model.layers)-3].output)
    

    

    list_probability = []
    #X = X[:100]
    # Get weights
    weights = [layer.get_weights() for layer in model.layers]
    output_weights = weights[len(weights)-2][0]
    temp = np.array(output_weights)
    temp = temp.flatten()
    output_weights = temp.tolist()

    for offset in range(0, int(math.floor(len(X)/100))):
        print("processing chunk " + str(offset+1) + "/" + str(int(math.floor(len(X)/100))))
        X2 = X[100*offset:100*(offset+1)]
        
        # Used maps to get output of hidden layers
        cnn_out = cnn_layer.predict(X2)
        max_pooling = max_pooling_layer.predict(X2)
        cnn_out_2 = cnn_layer_2.predict(X2)
        max_pooling_2 = max_pooling_layer_2.predict(X2)
        feature_vector = feature_vector_output.predict(X2)
        
        for input_index in range(0, len(X2)):
            #print("--ticket(" + str(input_index) + ")")
            #print("List length currently = " + str(len(list_probability))) 
            f1 = max_pooling[input_index][0][0].tolist()
            f2 = max_pooling_2[input_index][0][0].tolist()
            f3 = f1 + f2
            w = []
            for i in range(0, len(output_weights)):
                w.append(f3[i] * output_weights[i])
            
            prob = softmax(w)
            #print(prob.shape)
           
            
            for i in range(0,300): #Here i represents the filter(feature) number
                # Uni grams
                found = False
                
                max_pooling_top = list(zip(list(range(1, len(max_pooling[input_index][0][0]))), max_pooling[input_index][0][0]))
                #print(max_pooling_top)
                max_pooling_top = sorted(max_pooling_top, key=lambda x: (-x[1]))
                max_pooling_top = max_pooling_top[:5]
                valid_index = [i for i, j in max_pooling_top] 
                
                
                if i in valid_index:
                    f_val_uni = max_pooling[input_index][0][0][i]
                    for j in range(1,1000): #Here j represents the word number
                        search_value_uni = cnn_out[input_index][j][0][i]
                        if search_value_uni == f_val_uni and found == False:
                            list_probability = append_to_list(list_probability, (reverse_word_map[j], prob[i], w[i]))
                            found = True
                            
                        
                # Bi grams         
                found = False
                max_pooling_top = list(zip(list(range(1, len(max_pooling_2[input_index][0][0]))), max_pooling_2[input_index][0][0]))
                max_pooling_top = sorted(max_pooling_top, key=lambda x: (-x[1]))
                max_pooling_top = max_pooling_top[:5]
                valid_index = [i for i, j in max_pooling_top] 
                
                
                if i in valid_index:
                    f_val_bi = max_pooling_2[input_index][0][0][i]
                    for j in range(1,999): #Here j represents the word number
                        search_value_bi = cnn_out_2[input_index][j][0][i]
                        if search_value_bi == f_val_bi and found == False:
                            word = reverse_word_map[j] + " " + reverse_word_map[j+1]
                            list_probability = append_to_list(list_probability, (word, prob[300+i], w[300+i]))
                            found = True            
                     
        
    # Divide sum by amount to get average weighting of n-gram.
    for x in range(0,len(list_probability)):
         list_probability[x] = (list_probability[x][0], list_probability[x][1]/list_probability[x][3], list_probability[x][2]/list_probability[x][3], list_probability[x][3])
    
    print(list_probability[0])
    #list_print(list_probability)
       
    max_list = sorted(list_probability, key=lambda x: (-x[1]))
    print(len(list_probability))
    print("Max weighted words")
    list_print(max_list[:50])
    
    max_list = sorted(list_probability, key=lambda x: (-x[3]))
    print(len(list_probability))
    print("Max weighted common words")
    list_print(max_list[:50])

    
type = "general"    
extract_key_words(load_new('file.csv', amount = 10000, type = type), type)
#extract_key_words(load_from_file('technical_debt_dataset.csv', amount = 400), type = type)
#extract_key_words_2(load_new('file.csv', amount = 2000, type = type), "m2")
