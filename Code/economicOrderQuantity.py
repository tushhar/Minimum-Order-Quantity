import numpy as np
import pandas as pd
from keras.layers import Dense, Input
from keras.layers.core import Dropout
from keras.models import Model
from keras import losses
from keras.optimizers import Adam,SGD

train = pd.read_csv('/Data/minorder.csv')
Y_train = train.ix[:,5].values.astype('float32')
X_train = (train.ix[:,0:4].values).astype('float32')



# pre-processing: divide by max and substract mean
mean = np.std(X_train)
scale = np.max(X_train)
X_train -= mean
X_train /= scale
meany = np.std(Y_train)
scaley = np.max(Y_train)
Y_train -= mean
Y_train /= scale

print Y_train
print X_train

main_input = Input(shape=(4,), dtype='float32')

x = Dense(4, activation='linear')(main_input)
x= Dropout(0.33)(x)
x = Dense(2, activation='linear')(x)
x= Dropout(0.33)(x)
preds = Dense(1, activation='linear')(x)

model = Model(main_input, preds)
opt = SGD(lr=0.001)
# model.compile(loss='mean_squared_error',optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0001),metrics=['acc'])
model.compile(loss = "mean_squared_error", optimizer = opt)
print "\nTraining model..."
model.fit(X_train, Y_train,batch_size=1,epochs=50)
print "Done!"
print "Evaluating..."
eval = model.evaluate(X_train,Y_train)
print "\ntraining loss = %s" %eval
