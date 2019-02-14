import numpy as np
from HodaDatasetReader import read_hoda_dataset
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten


np.random.seed(123)
X_train, Y_train = read_hoda_dataset(dataset_path='./DigitDB/Train 60000.cdb',
                                     images_height=28,
                                     images_width=28,
                                     one_hot=True,
                                     reshape=False)

X_test, Y_test = read_hoda_dataset(dataset_path='./DigitDB/Test 20000.cdb',
                                   images_height=28,
                                   images_width=28,
                                   one_hot=True,
                                   reshape=False)
X_test = np.reshape(X_test, (20000, 28, 28, 1))
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPool2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=30, batch_size=32, validation_split=0.1)
loss, acc = model.evaluate(X_test, Y_test)
prediction = model.predict(X_test)
