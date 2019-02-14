import numpy as np
from HodaDatasetReader import read_hoda_dataset
from keras.models import Sequential
from keras.layers import Dense, Dropout


np.random.seed(123)
X_train, Y_train = read_hoda_dataset(dataset_path='./DigitDB/Train 60000.cdb',
                                     images_height=10,
                                     images_width=10,
                                     one_hot=True,
                                     reshape=True)
X_train = np.squeeze(X_train)

X_test, Y_test = read_hoda_dataset(dataset_path='./DigitDB/Test 20000.cdb',
                                   images_height=10,
                                   images_width=10,
                                   one_hot=True,
                                   reshape=True)

model = Sequential()
model.add(Dense(100, activation='relu', input_dim=100))
model.add(Dense(50, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=30, batch_size=32, validation_split=0.1)
loss, acc = model.evaluate(X_test, Y_test)
prediction = model.predict(X_test)
