import numpy as np
from HodaDatasetReader import read_hoda_dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

X_train, Y_train = read_hoda_dataset(dataset_path='./DigitDB/Train 60000.cdb',
                                     images_height=10,
                                     images_width=10,
                                     one_hot=False,
                                     reshape=True)
X_train = np.squeeze(X_train)

X_test, Y_test = read_hoda_dataset(dataset_path='./DigitDB/Test 20000.cdb',
                                   images_height=10,
                                   images_width=10,
                                   one_hot=False,
                                   reshape=True)
X_test = np.squeeze(X_test)

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, Y_train)
Y_pred = neigh.predict(X_test)
acc = accuracy_score(Y_test, Y_pred)
print(acc)
