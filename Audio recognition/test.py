import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
import lightgbm as lgb
import warnings
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss
import time
import os
warnings.filterwarnings('ignore')
import wavio
import librosa
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,LSTM,TimeDistributed,Bidirectional
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import to_categorical
from keras.callbacks import LearningRateScheduler
import sklearn

import pickle
fileHandle = open ( 'traindata_v2.txt' ,'rb')
X_train_c = pickle.load(fileHandle)
fileHandle.close()
fileHandle = open ( 'ydata_v2.txt' ,'rb')
y1_train_c = pickle.load(fileHandle)
fileHandle.close()
fileHandle = open ( 'testdata_v2.txt' ,'rb')
X_test_c = pickle.load(fileHandle)
fileHandle.close()

filename_list = ['bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'four', 'go', 'happy', 'house', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'wow', 'yes', 'zero']
#cnn多分类
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)
print(X_test.shape)
nfold = 5
kf = KFold(n_splits=nfold, shuffle=True, random_state=2020)
prediction1 = np.zeros((len(X_test),30 ))
i = 0
for train_index, valid_index in kf.split(features, labels):
    print("\nFold {}".format(i + 1))
    train_x, train_y = features[train_index],labels[train_index]
    val_x, val_y = features[valid_index],labels[valid_index]
    train_x = train_x.reshape(train_x.shape[0],train_x.shape[1],train_x.shape[2],1)
    val_x = val_x.reshape(val_x.shape[0],val_x.shape[1],val_x.shape[2],1)
    print(train_x.shape)
    print(val_x.shape)
    train_y = to_categorical(train_y,30)
    val_y = to_categorical(val_y,30)
    print(train_y.shape)
    print(val_y.shape)
    model = Sequential()
    model.add(Convolution2D(32, (3, 3), activation='relu',input_shape = train_x.shape[1:]))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    model.add(Convolution2D(32, (3, 3),  activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(30, activation='softmax'))
    model.compile(optimizer='Adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    model.summary(line_length=80)
    history = model.fit(train_x, train_y, epochs=35, batch_size=32, validation_data=(val_x, val_y))
    y1 = model.predict(X_test)
    prediction1 += ((y1)) / nfold
    i += 1
y_pred=[list(x).index(max(x)) for x in prediction1]
sub = pd.read_csv('submission.csv')
cnt = 0
result = [0 for i in range(6835)]
for i in range(6835):
    ss = sub.iloc[i]['file_name']
    for j in range(6835):
        if fl[j] == ss:
            result[i] = y_pred[j]
            cnt = cnt+1
print(cnt)
result1 = []
for i in range(len(result)):
    result1.append(filename_list[result[i]])
print(result1[0:10])
df = pd.DataFrame({'file_name':sub['file_name'],'label':result1})
now = time.strftime("%Y%m%d_%H%M%S",time.localtime(time.time()))
fname="submit_"+now+r".csv"
df.to_csv(fname, index=False)