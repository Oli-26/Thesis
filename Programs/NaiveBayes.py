import pandas as pd
from io import StringIO
import re

#Define file name 
address = 'technical_debt_dataset.csv'



from LoadData import load_from_file

# Load dataframe
df = load_from_file(address, amount = 20000)



# Conversion form classification to category id(Useful for deciphering output)
category_id_df = df[['classification', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'classification']].values)
df.head()
print(id_to_category)


# Plot count of different types
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,6))
df.groupby('classification').commenttext.count().plot.bar(ylim=0)
plt.show()


#Create learning model
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
X_train, X_test, y_train, y_test = train_test_split(df['commenttext'], df['category_id'], random_state = 14)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = MultinomialNB().fit(X_train_tfidf, y_train)



# Test examples
print(clf.predict(count_vect.transform(["This is poorly designed. We should change the char x to an int and then cast it later."])))


print(clf.predict(count_vect.transform(["// This is design debt"])))