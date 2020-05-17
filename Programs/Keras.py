from keras.models import Sequential
from keras import layers

from LoadData import load_from_file
from ModelTesting import split_data
from NLP import create_vectorizer
from sklearn.model_selection import train_test_split


number_of_examples = 50000

df = load_from_file('technical_debt_dataset.csv', amount = number_of_examples)
X_train, X_test, y_train, y_test = X_train, X_test, y_train, y_test = train_test_split(df['commenttext'], df['category_id'], random_state = 15, train_size = 0.25)


print("Distribution -")
print(df.groupby('classification').commenttext.count()/(df.shape[0]/100))
print("-------------\n")

tfidf = create_vectorizer(10)
tfidf.fit(X_train)


X_train_v = tfidf.transform(X_train)
X_test_v = tfidf.transform(X_test)
input_dim = X_train_v.shape[1]

model = Sequential()
model.add(layers.Dense(100, input_dim=input_dim, activation='selu'))
model.add(layers.Dense(6, activation='sigmoid'))


from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

from keras.optimizers import SGD
opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#opt = SGD(lr=0.01)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])
model.summary()


history = model.fit(X_train_v, y_train, epochs=100, verbose=False, validation_data=(X_test_v, y_test), batch_size=20)

loss, accuracy = model.evaluate(X_train_v, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test_v, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

#prediction = model.predict(X_test_v)
#print(prediction)

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