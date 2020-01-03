#!/usr/bin/env python
# coding: utf-8

# In[22]:


from google.colab import files
uploaded_1 = files.upload()
uploaded_2 = files.upload()


# In[147]:


import pandas as pd
import io
train = pd.read_csv(io.StringIO(uploaded_1['poker-hand-training-true.csv'].decode('utf-8')), header=None)
test = pd.read_csv(io.StringIO(uploaded_2['poker-hand-testing.csv'].decode('utf-8')), header=None)
train.head()
test.head()


# In[ ]:


from keras.models import Sequential
from keras.layers.core import Dense, Activation


# In[ ]:


def build_multilayer_perceptron():
    model = Sequential()
    model.add(Dense(128), input_shape=(10, ))
    model.add(Activation('relu'))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('sigmoid'))
    return model


# In[ ]:


model = build_multilayer_perceptron()


# In[148]:


from sklearn import preprocessing
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import normalize

train_Y = train[:][10]
test_Y = test[:][10]

X_train, X_test, Y_train, Y_test = train_test_split(train, train_Y,train_size=0.7)


X_train.drop(10,axis=1,inplace=True)
X_test.drop(10,axis=1,inplace=True)
X_train = normalize(X_train)
X_test = normalize(X_test)



print(X_train.shape,Y_train.shape,X_test.shape,Y_test.shape)


# In[ ]:


import xgboost as xgb
from sklearn.model_selection import GridSearchCV


# In[ ]:


clf = xgb.XGBClassifier()


# In[ ]:


clf_cv = GridSearchCV(clf, {'max_depth': [2,4,6], 'n_estimators': [50,100,200]}, verbose=1)


# In[152]:


print(X_train.shape,Y_train.shape)
clf_cv.fit(X_train, Y_train,verbose=1)


# In[155]:


# 改めて最適パラメータで学習
clf = xgb.XGBClassifier(**clf_cv.best_params_)
clf.fit(X_train, Y_train,verbose=1)

# 学習モデルの保存、読み込み
# import pickle
# pickle.dump(clf, open("model.pkl", "wb"))
# clf = pickle.load(open("model.pkl", "rb"))

# 学習モデルの評価
pred = clf.predict(X_test)
print (confusion_matrix(Y_test, pred))
print (classification_report(Y_test, pred))


# In[157]:


from sklearn.ensemble import RandomForestClassifier
RFCmodel = RandomForestClassifier(n_estimators=100,random_state=0)
RFCmodel.fit(X_train,Y_train)
pred = RFCmodel.predict(X_test)
print(RFCmodel.score(X_test,Y_test))

