import pandas as pd
from io import StringIO
import re
from TextCleaner import cleanText
import numpy as np

def load_csv(address):
    i = 0
    for_pd = StringIO()
    with open(address) as f:
        for line in f:
            if(not line[0] == ' '): 
                new_line = re.sub(r',', "___", line.rstrip(), count = 2)
                segments = new_line.split("___")
                if(not new_line == None):
                    try:
                        new_line = segments[0] + "___" + segments[1] + "___" + cleanText(segments[2])
                        print(new_line, file=for_pd)
                    except:
                        i = i + 1
                else:
                    i = i + 1
    print(str(i) + " rows failed to load \n")
    for_pd.seek(0)
    return for_pd

    
def load_from_file(filename, amount):
    address = filename
    # Special loading function to create a new seperator (due to the comments containing commas).
    data_stream = load_csv(address)
    # We skip rows that dont behave well with our delimiter. (3 for this dataset).
    df = pd.read_csv(data_stream, sep = "___", error_bad_lines=False, engine='python', nrows = amount)
    df.head()
    # Make column names neat and pretty.
    df.columns = ['project', 'classification', 'commenttext']
    # We want to remove any erroneous classifiation rows. Or rows with empty comments.
    words = ["DESIGN", "DEFECT", "WITHOUT_CLASSIFICATION", "IMPLEMENTATION", "TEST", "DOCUMENTATION"]
    df = df[pd.notnull(df['commenttext'])]
    df = df[df['classification'].isin(words)]
    # Create a numeric factorisation of categories for machine learning purposes.
    codes, uniques = df['classification'].factorize()
    df['category_id'] = codes
    #Print category types with the percent rate at which they occur.
    unique, counts = np.unique(df['category_id'], return_counts=True)
    print(dict(zip(uniques, counts*100/(len(df['category_id'])))))

    return df

def split_by_project(df):
    unique = np.unique(df['project'])
    return_structure = []
    for i in range(0, len(unique)):
        return_structure.append(df[df['project'] == unique[i]])
    return return_structure
    
def categorize(x):
    if x == 140 or x == 141 or x == 142: # arch
        return 0
    if x == 143 or x == 144 or x == 145: # build
        return 1
    if x >= 146 and x <= 152: # code
        return 2
    if x == 153: # defect
        return 3
    if x == 154: # design
        return 0
    if x == 155 or x == 156: # doc
        return 4
    if x == 157 or x == 158: # requirements
        return 5
    if x == 159 or x == 160 or x == 161: # rest
        return 6
    if x == 162: # none
        return 7
    
    
def load_new(filename, amount):
    address = filename
    df = pd.read_csv(address, sep = ',', quotechar = '"', error_bad_lines = False, nrows = amount)
    df.head()
    df = df[pd.notnull(df['label'])]
    
    # Clean comment strings.
    df['commenttext'] = df['text'].str.replace('[^\w\s]','') 
    df['commenttext'] = df['commenttext'].str.replace('[\n\t]',' ') 
    df['commenttext'] = df['commenttext'].str.lower()
    
    # Convert categories into numeric form using categorize function. We reduce the number of categories into larger, more general super classes.
    v = np.vectorize(categorize)
    df['category_id'] = v(df.label)
    
    # Print out percentage occurance of different type of classification.
    unique, counts = np.unique(df['category_id'], return_counts=True)
    target_names = ['arch', 'build', 'code', 'defect', 'doc', 'requirements', 'test', 'none']
    print(dict(zip(target_names, counts*100/(len(df['category_id'])))))
    return df
 
    