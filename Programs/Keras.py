from keras.models import Sequential
from keras import layers

from LoadData import load_from_file
from ModelTesting import split_data
from NLP import create_vectorizer
number_of_examples = 20000
verbose = False
    
df = load_from_file('technical_debt_dataset.csv', amount = number_of_examples)
X_train, X_test, y_train, y_test = split_data(df)


tfidf = create_vectorizer(10)
tfidf.fit(X_train)


X_train_v = tfidf.transform(X_train)
X_test_v = tfidf.transform(X_test)
input_dim = X_train_v.shape[1]

model = Sequential()
model.add(layers.Dense(50, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.summary()


history = model.fit(X_train_v, y_train, epochs=10, verbose=False, validation_data=(X_test_v, y_test), batch_size=20)

loss, accuracy = model.evaluate(X_train_v, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test_v, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))


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