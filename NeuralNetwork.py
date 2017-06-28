import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn import metrics

def Nural_Model():
    model = Sequential()
    model.add(Dense(units = 10, activation = 'sigmoid', input_shape=(92650, 55)))
    model.add(Dropout(0.2))
    model.add(Dense(input_dim=10, units=3, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    print('reading training and testing data...')
    train_x = np.loadtxt('Pre_Process_Train.csv', delimiter=',')
    train = np.loadtxt('dota2Train.csv', delimiter=',')
    train_y = train[:, 0]
    test_x = np.loadtxt('Pre_Process_Test.csv', delimiter=',')
    test = np.loadtxt('dota2Test.csv', delimiter=',')
    test_y = test[:, 0]

    estimater = KerasClassifier(build_fn=Nural_Model, nb_epoch=40, batch_size=256)
    estimater.fit(train_x, train_y)
    predict = estimater.predict(test_x)

    precision = metrics.precision_score(test_y, predict)
    recall = metrics.recall_score(test_y, predict)
    print('precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall))
    accuracy = metrics.accuracy_score(test_y, predict)
    print('accuracy: %.2f%%' % (100 * accuracy))