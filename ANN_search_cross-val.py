from numpy.core.fromnumeric import shape
from numpy.lib.financial import rate
from numpy.lib.function_base import angle
from sklearn.base import BaseEstimator
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks, layers, metrics, optimizers
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
import os
import pandas as pd
import numpy as np
import csv
from keras import backend as K
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers.core import Dropout
from sklearn.model_selection import train_test_split



os.chdir(r'C:\Users\juan_\Desktop\Tesis')
filename = 'ANN_1_16.xlsx'
sh_na = 'ANN_1_16'
file = pd.read_excel(filename, sheet_name=sh_na)

ankle_x=file['Ankle_x']
ankle_y=file['Ankle_y']
ankle_z=file['Ankle_z']
knee_x=file['Knee_x']
knee_y=file['Knee_y']
knee_z=file['Knee_z']
hip_x=file['Hip_x']
hip_y=file['Hip_y']
hip_z=file['Hip_z']
pelvis_x=file['Pelvis_pos_x']
pelvis_y=file['Pelvis_pos_y']
pelvis_z=file['Pelvis_pos_z']
pelvis_vel_x=file['Pelvis_vel_x']
pelvis_vel_y=file['Pelvis_vel_y']
pelvis_vel_z=file['Pelvis_vel_z']
pelvis_acc_x=file['Pelvis_acc_x']
pelvis_acc_y=file['Pelvis_acc_y']
pelvis_acc_z=file['Pelvis_acc_z']
pelvis_ang_vel_x=file['Pelvis_ang_vel_x']
pelvis_ang_vel_y=file['Pelvis_ang_vel_y']
pelvis_ang_vel_z=file['Pelvis_ang_vel_z']
pelvis_ang_acc_x=file['Pelvis_ang_acc_x']
pelvis_ang_acc_y=file['Pelvis_ang_acc_y']
pelvis_ang_acc_z=file['Pelvis_ang_acc_z']
grf_x=file['GRF_x']
grf_y=file['GRF_y']
grf_z=file['GRF_z']

ankle_x_lis=ankle_x.tolist()
ankle_y_lis=ankle_y.tolist()
ankle_z_lis=ankle_z.tolist()
knee_x_lis=knee_x.tolist()
knee_y_lis=knee_y.tolist()
knee_z_lis=knee_z.tolist()
hip_x_lis=hip_x.tolist()
hip_y_lis=hip_y.tolist()
hip_z_lis=hip_z.tolist()
pelvis_x_lis=pelvis_x.tolist()
pelvis_y_lis=pelvis_y.tolist()
pelvis_z_lis=pelvis_z.tolist()
pelvis_vel_x_lis=pelvis_vel_x.tolist()
pelvis_vel_y_lis=pelvis_vel_y.tolist()
pelvis_vel_z_lis=pelvis_vel_z.tolist()
pelvis_acc_x_lis=pelvis_acc_x.tolist()
pelvis_acc_y_lis=pelvis_acc_y.tolist()
pelvis_acc_z_lis=pelvis_acc_z.tolist()
pelvis_ang_vel_x_lis=pelvis_ang_vel_x.tolist()
pelvis_ang_vel_y_lis=pelvis_ang_vel_y.tolist()
pelvis_ang_vel_z_lis=pelvis_ang_vel_z.tolist()
pelvis_ang_acc_x_lis=pelvis_ang_acc_x.tolist()
pelvis_ang_acc_y_lis=pelvis_ang_acc_y.tolist()
pelvis_ang_acc_z_lis=pelvis_ang_acc_z.tolist()
grf_x_lis=grf_x.tolist()
grf_y_lis=grf_y.tolist()
grf_z_lis=grf_z.tolist()

matrix=[]
for i in range(0, 16):
    x_data_full=[]
    y_data_full=[]
    x_data_out=[]
    y_Xdata_out=[]
    y_Ydata_out=[]
    y_Zdata_out=[]
    for j in range(0, 16160):
        if (j<i*1010) or (j>=(i*1010)+1010):
            x_data_full.append([ankle_x_lis[j], ankle_y_lis[j], ankle_z_lis[j], knee_x_lis[j], knee_y_lis[j], knee_z_lis[j], hip_x_lis[j], hip_y_lis[j], hip_z_lis[j]])
                                #pelvis_x_lis[j], pelvis_y_lis[j], pelvis_z_lis[j],
                                #pelvis_vel_x_lis[j], pelvis_vel_y_lis[j], pelvis_vel_z_lis[j], 
                                #pelvis_acc_x_lis[j], pelvis_acc_y_lis[j], pelvis_acc_z_lis[j],
                                #pelvis_ang_vel_x_lis[j], pelvis_ang_vel_y_lis[j], pelvis_ang_vel_z_lis[j],
                                #pelvis_ang_acc_x_lis[j], pelvis_ang_acc_y_lis[j], pelvis_ang_acc_z_lis[j]])
            y_data_full.append([grf_x_lis[j], grf_y_lis[j], grf_z_lis[j]])

        if (j>=i*1010) and (j<(i*1010)+1010):
            x_data_out.append([ankle_x_lis[j], ankle_y_lis[j], ankle_z_lis[j], knee_x_lis[j], knee_y_lis[j], knee_z_lis[j], hip_x_lis[j], hip_y_lis[j], hip_z_lis[j]])
                                #pelvis_x_lis[j], pelvis_y_lis[j], pelvis_z_lis[j],
                                #pelvis_vel_x_lis[j], pelvis_vel_y_lis[j], pelvis_vel_z_lis[j], 
                                #pelvis_acc_x_lis[j], pelvis_acc_y_lis[j], pelvis_acc_z_lis[j],
                                #pelvis_ang_vel_x_lis[j], pelvis_ang_vel_y_lis[j], pelvis_ang_vel_z_lis[j],
                                #pelvis_ang_acc_x_lis[j], pelvis_ang_acc_y_lis[j], pelvis_ang_acc_z_lis[j]])

            y_Xdata_out.append(grf_x_lis[j])
            y_Ydata_out.append(grf_y_lis[j])
            y_Zdata_out.append(grf_z_lis[j])


    for j in range(0, 15):
        x_data = x_data_full[1010*j:1010*(j+1)]
        y_data = y_data_full[1010*j:1010*(j+1)]
        if j==0:
            x_train, x_test = train_test_split(x_data, test_size=0.1, random_state=42)
            x_train, x_valid = train_test_split(x_train, test_size=0.2, random_state=42)

            y_train, y_test = train_test_split(y_data, test_size=0.1, random_state=42)
            y_train, y_valid = train_test_split(y_train, test_size=0.2, random_state=42)
        else:
            x_train_, x_test_ = train_test_split(x_data, test_size=0.1, random_state=42)
            x_train_, x_valid_ = train_test_split(x_train_, test_size=0.2, random_state=42)

            y_train_, y_test_ = train_test_split(y_data, test_size=0.1, random_state=42)
            y_train_, y_valid_ = train_test_split(y_train_, test_size=0.2, random_state=42)

            x_train = x_train + x_train_
            x_valid = x_valid + x_valid_
            x_test = x_test + x_test_

            y_train = y_train + y_train_
            y_valid = y_valid + y_valid_
            y_test = y_test + y_test_


    x_train = np.array(x_train)
    x_valid = np.array(x_valid)
    y_train = np.array(y_train)
    y_valid = np.array(y_valid)

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_valid = scaler.transform(x_valid)

    def rmse(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred-y_true), axis=-1))

    def build_model(n_layers, neurons, dropout):
        lr_ini = 0.1
        lr_end = 0.0001
        b_s = 101
        epo = 500
        steps_epoch = int(len(x_train)/b_s)
        decay = (lr_end/lr_ini)**(1/epo)
        lr_sch = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=lr_end,
            decay_steps=steps_epoch,
            decay_rate=decay,
            staircase=True)
        model = keras.models.Sequential()
        model.add(keras.layers.InputLayer(input_shape=x_train.shape[1:]))
        for l in range(n_layers):
            model.add(keras.layers.Dropout(rate=dropout))
            model.add(keras.layers.Dense(neurons, activation='elu', kernel_initializer='he_normal'))
        model.add(keras.layers.Dense(3))
        opt = tf.keras.optimizers.Adam(learning_rate=lr_sch)
        model.compile(loss=rmse, optimizer=opt)
        return model

    keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model)

    hyperparam = {
        'n_layers': [1, 2, 3],
        'neurons': [250, 500, 750],
        'dropout': [0.025, 0.05, 0.075],
    }

    search = GridSearchCV(keras_reg, hyperparam, cv=5)
    search.fit(x_train, y_train, batch_size=101, epochs=500,
            validation_data=(x_valid, y_valid),
            callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)])
    
    matrix.append(str(i+1)+'_out')
    matrix.append(search.best_params_)
    matrix.append(search.best_score_)
    final_model = search.best_estimator_.model
    final_model.save('Angle_inputs_'+str(i+1)+'_out'+'.h5')

matrix = np.array(matrix)

with open(r'C:\Users\juan_\Desktop\Tesis\Grid_Search\\' + 'Angles_Inputs.csv', 'x', newline='') as f:
    writer = csv.writer(f)      
    writer.writerows(zip(matrix))
