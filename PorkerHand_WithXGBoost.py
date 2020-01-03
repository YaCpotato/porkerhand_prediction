#!/usr/bin/env python
# coding: utf-8

# In[156]:


from google.colab import files
uploaded_1 = files.upload()
uploaded_2 = files.upload()


# In[ ]:


import pandas as pd
import io
train = pd.read_csv(io.StringIO(uploaded_1['poker-hand-training-true.csv'].decode('utf-8')), header=None)
test = pd.read_csv(io.StringIO(uploaded_2['poker-hand-testing.csv'].decode('utf-8')), header=None)
train.head()
test.head()


# In[ ]:


from sklearn import preprocessing
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import normalize
train_Y = np_utils.to_categorical((train[:][10])
test_Y = np_utils.to_categorical(test[:][10])

X_train, X_test, Y_train, Y_test = train_test_split(train, train_Y,train_size=0.7)


X_train.drop(10,axis=1,inplace=True)
X_test.drop(10,axis=1,inplace=True)
X_train = normalize(X_train)
X_test = normalize(X_test)



print(X_train.shape,Y_train.shape,X_test.shape,Y_test.shape)


# In[ ]:


import time
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

    # モデル訓練
start = time.time()
result = model.fit(X_train, Y_train, nb_epoch=50,batch_size=1,verbose=1,validation_split=0.2)
end=time.time() - start
print('処理時間')
print(end)
loss, accuracy = model.evaluate(X_test,Y_test,verbose=2)
print("Accuracy = {:.2f}".format(accuracy))
pred = model.predict(X_test)
pred = np.argmax(pred, axis=1)
Y = np.argmax(Y_test,axis=1)
print(confusion_matrix(Y,pred))

