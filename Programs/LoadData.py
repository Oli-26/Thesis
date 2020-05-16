import pandas as pd
from io import StringIO
import re
from TextCleaner import cleanText

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
    
    
    return df
    