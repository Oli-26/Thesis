import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
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


import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

def list_print(input_list):
    for x in range(0,len(input_list)):
        print("[" + str(x) + "] " + str(input_list[x][0]) + "(" + str(input_list[x][1]) + ") = " + str(input_list[x][2]*100) + "%")
    

def group_list(input_list):
    grouped_list = []
    for x in range(0,len(input_list)):
        found = False
        for y in range(0, len(grouped_list)):
            if(input_list[x][0] == grouped_list[y][0]):
                grouped_list[y] = (grouped_list[y][0], grouped_list[y][1]+1, grouped_list[y][2]+input_list[x][1])
                found = True
        if found == False:
            grouped_list.append((input_list[x][0], 1, input_list[x][1]))
    for y in range(0, len(grouped_list)):
        grouped_list[y] = (grouped_list[y][0], grouped_list[y][1], grouped_list[y][2]/grouped_list[y][1])
    return grouped_list
    

def extract_key_words(df, type):
    print("extracting key words for " + type)
    K.clear_session()
    
    # Load model 
    model = keras.models.load_model('m2', custom_objects={'get_f1':get_f1})
    
    # We feed a comment into a trained CNN model to obtain the feature vector X in R(mX1) of this comment in the fully connected layer. Each feature x_i in the feature vector X corresponds to a filter.
    tokenizer = RegexpTokenizer(r'\w+')
    
    # tokenize dataset
    processed_comments = []
    for comment in df['commenttext']:
        tokens = tokenizer.tokenize(comment)
        processed_comments.append(tokens)
    tokenizer = Tokenizer(num_words=None,filters = '!##$%&()*+', lower = True, split = ' ')
    tokenizer.fit_on_texts(processed_comments)
    X = tokenizer.texts_to_sequences(processed_comments)
    X = pad_sequences(X, 1000)
    Y = df['category_id']
    
    # get predictions
    pred_Y = model.predict(X)
    pred_Y = np.argmax(pred_Y, axis=1)
    
    newX = X
    
    # Create reverse word map
    word_index = tokenizer.word_index
    reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
    
    # Create map between input and hidden layers
    cnn_layer = Model(model.input, model.layers[len(model.layers)-9].output) # uni
    max_pooling_layer = Model(model.input, model.layers[len(model.layers)-7].output) # uni
    
    cnn_layer_2 = Model(model.input, model.layers[len(model.layers)-8].output) # bi
    max_pooling_layer_2 = Model(model.input, model.layers[len(model.layers)-6].output) # bi
    
    feature_vector_output = Model(model.input, model.layers[len(model.layers)-3].output)
    
    # Used maps to get output of hidden layers
    cnn_out = cnn_layer.predict(newX)
    max_pooling = max_pooling_layer.predict(newX)
    cnn_out_2 = cnn_layer_2.predict(newX)
    max_pooling_2 = max_pooling_layer_2.predict(newX)
    feature_vector = feature_vector_output.predict(newX)

    tallyMax = []
    for input_index in range(0, len(X)):
        words_list = []
        # Uni grams
        for i in range(0,300): #Here i represents the filter(feature) number
            # Search for feature
            found = False
            f_val = max_pooling[input_index][0][i]
            for j in range(1,1000): #Here j represents the word number
                search_value = cnn_out[input_index][j][i]
                if search_value == f_val and found == False:
                    words_list.append(reverse_word_map[j])
                    found = True
        # Bi grams            
        for i in range(0,300): #Here i represents the filter(feature) number
            # Search for feature
            found = False
            f_val = max_pooling_2[input_index][0][i]
            for j in range(1,999): #Here j represents the word number
                search_value = cnn_out_2[input_index][j][i]
                if search_value == f_val and found == False:
                    words_list.append(reverse_word_map[j] + " " + reverse_word_map[j+1])
                    found = True            

        # Get weights
        weights = [layer.get_weights() for layer in model.layers]
        output_weights = weights[len(weights)-2]
        
        # Get combination of feature vector and feaure weights.
        product_weights_features = []
        for i in range(0,599):
            product_weights_features.append(output_weights[0][i][0]*feature_vector[input_index][i])
        
        weightings_words = list(zip(words_list, product_weights_features)) # zip feature weights with the words they correspond to.
        weightings_words = group_list(weightings_words) # Group identical words, take average value as weighting
        weightings_words = sorted(weightings_words, key=lambda x: (-x[2])) # Sort by third element, largest first.
        
        # Add results to global store.
        found = False
        for word_tuple in weightings_words:
            for x in range(0,len(tallyMax)):
                if tallyMax[x][0] == word_tuple[0]:
                    tallyMax[x] = (tallyMax[x][0],tallyMax[x][1]+word_tuple[1], tallyMax[x][2]+word_tuple[2])
                    found = True
            if(found == False):
                tallyMax.append((word_tuple[0], word_tuple[1], word_tuple[2]))
     
    # Divide sum by amount to get average weighting of n-gram.
    for x in range(0,len(tallyMax)):
         tallyMax[x] = (tallyMax[x][0], tallyMax[x][1], tallyMax[x][2]/tallyMax[x][1])

    filtered_tally = []
    #for x in range(0,len(tallyMax)):
         #if(tallyMax[x][1] > 100):
            #filtered_tally.append(tallyMax[x])
    filtered_tally = tallyMax        
    tallyMax = sorted(filtered_tally, key=lambda x: (-x[2]))
    tallyMin = sorted(filtered_tally, key=lambda x: (x[2]))
    
    print("Max weighted words")
    list_print(tallyMax[:10])
    print("\nMin weighted words")
    list_print(tallyMin[:10])
    
    tallyMax = sorted(filtered_tally, key=lambda x: (-x[1]))
    print("\nMax weighted words")
    list_print(tallyMax[:10])
    
type = "general"    
#extract_key_words(load_new('file.csv', amount = 2000, type = type), type)
extract_key_words(load_from_file('technical_debt_dataset.csv', amount = 2000), type = type)
