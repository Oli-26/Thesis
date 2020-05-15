import pandas as pd
from io import StringIO
import re
from TextCleaner import cleanText

def load_csv(address):
    
    for_pd = StringIO()
    with open(address) as f:
        for line in f:
            if(not line[0] == ' '): 
                new_line = re.sub(r',', "___", line.rstrip(), count = 2)
                if(not new_line == None):
                    print(new_line, file=for_pd)

    for_pd.seek(0)
    return for_pd

    
def load_from_file(filename, amount):
    address = filename #'technical_debt_dataset.csv'
    
    data_stream = load_csv(address)
    
    # We skip rows that dont behave well with our delimiter. (3 for this dataset).
    df = pd.read_csv(data_stream, sep = "___", error_bad_lines=False, engine='python', nrows = amount)

    df.head()

    words = ["DESIGN", "DEFECT", "WITHOUT_CLASSIFICATION", "IMPLEMENTATION", "TEST", "DOCUMENTATION"]

    col = ['classification', 'commenttext']
    df.columns = ['project', 'classification', 'commenttext']
    df = df[col]
    df = df[pd.notnull(df['commenttext'])]

    df = df[df['classification'].isin(words)]
    df.columns = ['classification', 'commenttext']
    df['category_id'] = df['classification'].factorize()[0]
    
   

    #df = df.apply(lambda x: cleanText(x) if x.name == 'commenttext' else x)
    
    return df
    