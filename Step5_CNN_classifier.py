"""
    Copyright 2017-2022 Department of Electrical and Computer Engineering
    University of Houston, TX/USA
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
    Please contact Akshay Sujatha Ravindran for more info about licensing asujatharavindran@uh.edu
    via github issues section.
    **********************************************************************************
    Author:     Akshay Sujatha Ravindran
    File:       Step5_CNN_classifier
    Comments:   This is the main file to train a convolutional network to predict
    emotional states from PPG derived RR-interval based features (Point process modeled)
    **********************************************************************************
"""


# Load the libraries
from keras.layers import Dense, Dropout,Input, Conv2D,BatchNormalization,Activation,MaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.utils import to_categorical
import scipy.io as sio
import os
from sklearn.utils import class_weight
import tensorflow as tf
from keras import backend as K
import gc
from sklearn.metrics import f1_score, precision_score, recall_score

np.random.seed(100) 
HIST_VLoss,HIST_TLoss,HIST_VAcc,HIST_TAcc, pred, SUB, P_score, F_score,R_score,K_val,Len = [],[],[],[],[],[],[],[],[],[],[]

for iterate in range(5):    
    seed=np.random.randint(1,200)    # set the seed constant for each fold
    for j in range(3): # each class
        for i in range(22): # for each subjects
                training=sio.loadmat('Data_Class_%d_Subject_%d.mat'%((j+1),(i+1)))
                X=training['Feature']
                Y=np.ravel(training['labels'])
                identifier_val=training['identifier_val']
                
                
                
                X= np.reshape(X, (X.shape[0], X.shape[1], X.shape[2],1))                
                loc=np.where(np.isin(identifier_val[:,1],i+1))[0]
                
                
                # Shuffle unique trials to divide into training and validation set
                loc0=loc[np.where(Y[loc]==0)[0]]
                loc1=loc[np.where(Y[loc]==1)[0]]  
                classes=identifier_val[:,0]
                class0=np.unique(classes[loc0])
                class1=np.unique(classes[loc1])    
                
                
                order = np.arange(len(class0))
                np.random.seed(seed) 
                np.random.shuffle(order)
                class0=class0[order]
                
                order = np.arange(len(class1))
                np.random.seed(seed) 
                np.random.shuffle(order)
                class1=class1[order]
                
                
                # Divide 20% trials into validation set and rest into training set
                classes=5
                Test= np.append(class0[:classes], class1[:classes])
                Train= np.append(class0[classes:], class1[classes:])    

                test_loc=np.where(np.logical_and(np.isin(identifier_val[:,0],Test), np.isin(identifier_val[:,1],i+1)))[0]
                train_loc=np.where(np.logical_and(np.isin(identifier_val[:,0],Train), np.isin(identifier_val[:,1],i+1)))[0]
                
                
                # Shuffle the data
                order = np.arange(len(test_loc))
                np.random.seed(seed) 
                np.random.shuffle(order)
                test_loc=test_loc[order]                
                
                
                order = np.arange(len(train_loc))
                np.random.seed(seed) 
                np.random.shuffle(order)
                train_loc=train_loc[order]
                
                
                # To account for class imbalance
                class_weights = class_weight.compute_class_weight('balanced',
                                                     np.unique(Y[train_loc]),
                                                     Y[train_loc])
                class_weights = {i : class_weights[i] for i in range(2)}
                # Divide into training and validation set 
                X_train=X[train_loc]
                X_valid=X[test_loc]
                y_train=Y[train_loc]
                y_valid=Y[test_loc]
                Y_train=to_categorical(y_train)
                Y_valid=to_categorical(y_valid)
                
                #%% Define model architecture   
                ch = 6 # Number of features
                act='tanh' # Activation function
                dense_num_units,cnn_units = 16, 8 # Model architecture hyper parameters
                bn_axis = 3 # batch normalization dimension
                
                
                input_shape = (320, ch,1)     
                Allinput_img = Input(shape=(input_shape))  
                x_c = Conv2D(cnn_units,kernel_size=(2,1), strides=(2, 1), name='convch_0',use_bias=True)(Allinput_img) 
                x_c = BatchNormalization(axis = 3)(x_c)                
                x_c = Activation(act)(x_c)
                x_c = MaxPooling2D(pool_size=(2, 1))(x_c)
        
                x_c = Conv2D(cnn_units,kernel_size=(2,1) , strides=(2, 1) ,use_bias=True)(x_c)                
                x_c = BatchNormalization(axis = 3)(x_c)
                x_c = Activation(act)(x_c)   
                x_c = MaxPooling2D(pool_size=(2, 1))(x_c)
        
                x_c = Conv2D(cnn_units,kernel_size=(2,ch) , strides=(2, 1) ,use_bias=True)(x_c)                
                x_c = BatchNormalization(axis = 3)(x_c)
                x_c = Activation(act)(x_c)   
                x_c = MaxPooling2D(pool_size=(2, 1))(x_c) 
                
                
                x_c=GlobalAveragePooling2D()(x_c)       
                x=(Dropout(rate = 0.5,name='Drop_D2'))(x_c)
                Out=(Dense(2,activation='softmax' ,name='Allenc_18'))(x)                
                model=Model(Allinput_img,Out)
                model.compile(optimizer= Adam(lr=0.000001), loss='categorical_crossentropy',metrics=['accuracy'])   
                model.summary()
               
               
                Checkpoint_filename = './save_weights_CNN_RR.hdf5' # File to save the model weights to    
                
                # Callbacks to perfrorm early stopping if model does not improve in successive 5 epochs and to save 
                # model weights only if there is improvement
                callback_array = [EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto'),
                ModelCheckpoint(Checkpoint_filename, monitor='val_loss', verbose=1, save_best_only=True, mode='auto') ] 

                # Train the model
                history=model.fit(X_train, Y_train, 
                                  epochs=200,validation_data=(X_valid, Y_valid),
                                  batch_size=32, verbose=1, callbacks=callback_array, shuffle=True, class_weight=class_weights) 
                model.load_weights(Checkpoint_filename)
                print('Iteration number: %d; Condition: %d; Subject: %d'%(iterate,j,i))


                # Compute the performance metrics
                y_pred = np.argmax(model.predict(X_valid),axis=1)
                F_score.append([f1_score(y_valid, y_pred, average="macro"),f1_score(y_valid, y_pred, average="micro"),f1_score(y_valid, y_pred, average="weighted")])
                P_score.append([precision_score(y_valid, y_pred, average="macro"),precision_score(y_valid, y_pred, average="micro"),precision_score(y_valid, y_pred, average="weighted")])
                R_score.append([recall_score(y_valid, y_pred, average="macro"),recall_score(y_valid, y_pred, average="micro"),recall_score(y_valid, y_pred, average="weighted")])
                    
                
                HIST_VLoss.append(history.history['val_loss'])
                HIST_TLoss.append(history.history['loss'])
                HIST_VAcc.append(history.history['val_accuracy'])
                HIST_TAcc.append(history.history['accuracy'])
                pred.append(model.evaluate(X_valid,Y_valid))
                SUB.append(i)
                Len.append(len(history.history['val_loss']))
                
                # Reset the variables and delete the model
                os.remove(Checkpoint_filename)
                K.clear_session()
                gc.collect()
                del model, training, X ,Y, identifier_val, loc,loc0,loc1,order, class0, class1, Test, Train, test_loc, train_loc,class_weights, X_train, X_valid, Y_valid,Y_train
    
        
    # Save the final results
    sio.savemat('Prediction_V%d.mat'%(iterate), {'HIST_VLoss':HIST_VLoss,
                                'HIST_VAcc':HIST_VAcc,'HIST_TLoss':HIST_TLoss,
                                'HIST_TAcc':HIST_TAcc,'pred':pred,'sub':SUB,
                                'F_score':F_score,'P_score':P_score,'R_score':R_score,'Length':Len,}) 
    
