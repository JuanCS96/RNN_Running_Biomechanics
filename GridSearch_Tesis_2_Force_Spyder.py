from numpy.core.fromnumeric import shape
from numpy.lib.function_base import angle
from sklearn.base import BaseEstimator
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks, layers, metrics, optimizers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import csv
import scipy as sp
from scipy.stats import pearsonr
from scipy.interpolate import interp1d
from sklearn.model_selection import GridSearchCV
from keras import backend as K
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers.core import Dropout
from sklearn.model_selection import train_test_split
import socket

hostname=socket.gethostname();
if hostname=='PC64926':
    os.chdir(r'D:\JuanCordero\NN_GRF_M_prediction_running')
else:
    os.chdir(r'C:\Users\juan_\Desktop\Tesis\Tesis_2')
filename='ANN_Inputs_1_16.xlsx'
sh_na='ANN_Inputs_1_16'
file=pd.read_excel(filename, sheet_name=sh_na)

ankle_x=file['Ankle_x']
ankle_y=file['Ankle_y']
ankle_z=file['Ankle_z']
knee_x=file['Knee_x']
knee_y=file['Knee_y']
knee_z=file['Knee_z']
hip_x=file['Hip_x']
hip_y=file['Hip_y']
hip_z=file['Hip_z']
ankle_angular_vel_x=file['Ankle_Angular_Vel_x']
ankle_angular_vel_y=file['Ankle_Angular_Vel_y']
ankle_angular_vel_z=file['Ankle_Angular_Vel_z']
knee_angular_vel_x=file['Knee_Angular_Vel_x']
knee_angular_vel_y=file['Knee_Angular_Vel_y']
knee_angular_vel_z=file['Knee_Angular_Vel_z']
hip_angular_vel_x=file['Hip_Angular_Vel_x']
hip_angular_vel_y=file['Hip_Angular_Vel_y']
hip_angular_vel_z=file['Hip_Angular_Vel_z']
ankle_angular_acc_x=file['Ankle_Angular_Acc_x']
ankle_angular_acc_y=file['Ankle_Angular_Acc_y']
ankle_angular_acc_z=file['Ankle_Angular_Acc_z']
knee_angular_acc_x=file['Knee_Angular_Acc_x']
knee_angular_acc_y=file['Knee_Angular_Acc_y']
knee_angular_acc_z=file['Knee_Angular_Acc_z']
hip_angular_acc_x=file['Hip_Angular_Acc_x']
hip_angular_acc_y=file['Hip_Angular_Acc_y']
hip_angular_acc_z=file['Hip_Angular_Acc_z']
pelvis_lineal_vel_x=file['Pelvis_Lineal_Vel_x']
pelvis_lineal_vel_y=file['Pelvis_Lineal_Vel_y']
pelvis_lineal_vel_z=file['Pelvis_Lineal_Vel_z']
pelvis_lineal_acc_x=file['Pelvis_Lineal_Acc_x']
pelvis_lineal_acc_y=file['Pelvis_Lineal_Acc_y']
pelvis_lineal_acc_z=file['Pelvis_Lineal_Acc_z']
pelvis_angular_vel_x=file['Pelvis_Angular_Vel_x']
pelvis_angular_vel_y=file['Pelvis_Angular_Vel_y']
pelvis_angular_vel_z=file['Pelvis_Angular_Vel_z']
pelvis_angular_acc_x=file['Pelvis_Angular_Acc_x']
pelvis_angular_acc_y=file['Pelvis_Angular_Acc_y']
pelvis_angular_acc_z=file['Pelvis_Angular_Acc_z']
grf_x=file['GRF_x_15']
grf_y=file['GRF_y_15']
grf_z=file['GRF_z_15']

ankle_x=ankle_x.tolist()
ankle_y=ankle_y.tolist()
ankle_z=ankle_z.tolist()
knee_x=knee_x.tolist()
knee_y=knee_y.tolist()
knee_z=knee_z.tolist()
hip_x=hip_x.tolist()
hip_y=hip_y.tolist()
hip_z=hip_z.tolist()
ankle_angular_vel_x=ankle_angular_vel_x.tolist()
ankle_angular_vel_y=ankle_angular_vel_y.tolist()
ankle_angular_vel_z=ankle_angular_vel_z.tolist()
knee_angular_vel_x=knee_angular_vel_x.tolist()
knee_angular_vel_y=knee_angular_vel_y.tolist()
knee_angular_vel_z=knee_angular_vel_z.tolist()
hip_angular_vel_x=hip_angular_vel_x.tolist()
hip_angular_vel_y=hip_angular_vel_y.tolist()
hip_angular_vel_z=hip_angular_vel_z.tolist()
ankle_angular_acc_x=ankle_angular_acc_x.tolist()
ankle_angular_acc_y=ankle_angular_acc_y.tolist()
ankle_angular_acc_z=ankle_angular_acc_z.tolist()
knee_angular_acc_x=knee_angular_acc_x.tolist()
knee_angular_acc_y=knee_angular_acc_y.tolist()
knee_angular_acc_z=knee_angular_acc_z.tolist()
hip_angular_acc_x=hip_angular_acc_x.tolist()
hip_angular_acc_y=hip_angular_acc_y.tolist()
hip_angular_acc_z=hip_angular_acc_z.tolist()
pelvis_lineal_vel_x=pelvis_lineal_vel_x.tolist()
pelvis_lineal_vel_y=pelvis_lineal_vel_y.tolist()
pelvis_lineal_vel_z=pelvis_lineal_vel_z.tolist()
pelvis_lineal_acc_x=pelvis_lineal_acc_x.tolist()
pelvis_lineal_acc_y=pelvis_lineal_acc_y.tolist()
pelvis_lineal_acc_z=pelvis_lineal_acc_z.tolist()
pelvis_angular_vel_x=pelvis_angular_vel_x.tolist()
pelvis_angular_vel_y=pelvis_angular_vel_y.tolist()
pelvis_angular_vel_z=pelvis_angular_vel_z.tolist()
pelvis_angular_acc_x=pelvis_angular_acc_x.tolist()
pelvis_angular_acc_y=pelvis_angular_acc_y.tolist()
pelvis_angular_acc_z=pelvis_angular_acc_z.tolist()
grf_x=grf_x.tolist()
grf_y=grf_y.tolist()
grf_z=grf_z.tolist()

matrix=[]
for i in range(0, 16):
    x_data_full=[]
    y_data_full=[]
    x_data_out=[]
    y_Xdata_out=[]
    y_Ydata_out=[]
    y_Zdata_out=[]
    #y_hip_flex_m_out=[]
    #y_hip_abd_m_out=[]
    #y_hip_rot_m_out=[]
    #y_knee_flex_m_out=[]
    #y_ank_flex_m_out=[]
    #y_ank_abd_m_out=[]
    for j in range(0, 16160):
        if (j<i*1010) or (j>=(i*1010)+1010):
            x_data_full.append([ankle_x[j], ankle_y[j], ankle_z[j], knee_x[j], knee_y[j], knee_z[j], hip_x[j], hip_y[j], hip_z[j],
                                #ankle_angular_vel_x[j], ankle_angular_vel_y[j], ankle_angular_vel_z[j],
                                ankle_angular_acc_x[j], ankle_angular_acc_y[j], ankle_angular_acc_z[j],
                                #knee_angular_vel_x[j], knee_angular_vel_y[j], knee_angular_vel_z[j],
                                knee_angular_acc_x[j], knee_angular_acc_y[j], knee_angular_acc_z[j], 
                                #hip_angular_vel_x[j], hip_angular_vel_y[j], hip_angular_vel_z[j],
                                hip_angular_acc_x[j], hip_angular_acc_y[j], hip_angular_acc_z[j],
                                #pelvis_lineal_vel_x[j], pelvis_lineal_vel_y[j], pelvis_lineal_vel_z[j],
                                pelvis_lineal_acc_x[j], pelvis_lineal_acc_y[j], pelvis_lineal_acc_z[j], 
                                #pelvis_angular_vel_x[j], pelvis_angular_vel_y[j], pelvis_angular_vel_z[j],
                                pelvis_angular_acc_x[j], pelvis_angular_acc_y[j], pelvis_angular_acc_z[j]])
            y_data_full.append([grf_x[j], grf_y[j], grf_z[j]])
            #y_data_full.append([hip_flex_m[j], hip_abd_m[j], hip_rot_m[j], knee_flex_m[j], ank_flex_m[j], ank_abd_m[j]])

        if (j>=i*1010) and (j<(i*1010)+1010):
            x_data_out.append([ankle_x[j], ankle_y[j], ankle_z[j], knee_x[j], knee_y[j], knee_z[j], hip_x[j], hip_y[j], hip_z[j],
                                #ankle_angular_vel_x[j], ankle_angular_vel_y[j], ankle_angular_vel_z[j],
                                ankle_angular_acc_x[j], ankle_angular_acc_y[j], ankle_angular_acc_z[j],
                                #knee_angular_vel_x[j], knee_angular_vel_y[j], knee_angular_vel_z[j],
                                knee_angular_acc_x[j], knee_angular_acc_y[j], knee_angular_acc_z[j], 
                                #hip_angular_vel_x[j], hip_angular_vel_y[j], hip_angular_vel_z[j],
                                hip_angular_acc_x[j], hip_angular_acc_y[j], hip_angular_acc_z[j],
                                #pelvis_lineal_vel_x[j], pelvis_lineal_vel_y[j], pelvis_lineal_vel_z[j],
                                pelvis_lineal_acc_x[j], pelvis_lineal_acc_y[j], pelvis_lineal_acc_z[j], 
                                #pelvis_angular_vel_x[j], pelvis_angular_vel_y[j], pelvis_angular_vel_z[j],
                                pelvis_angular_acc_x[j], pelvis_angular_acc_y[j], pelvis_angular_acc_z[j]])

            y_Xdata_out.append(grf_x[j])
            y_Ydata_out.append(grf_y[j])
            y_Zdata_out.append(grf_z[j])
            #y_hip_flex_m_out.append(hip_flex_m[j])
            #y_hip_abd_m_out.append(hip_abd_m[j])
            #y_hip_rot_m_out.append(hip_rot_m[j])
            #y_knee_flex_m_out.append(knee_flex_m[j])
            #y_ank_flex_m_out.append(ank_flex_m[j])
            #y_ank_abd_m_out.append(ank_abd_m[j])


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
        'dropout': [0, 0.025, 0.05, 0.075],
    }

    search = GridSearchCV(keras_reg, hyperparam, cv=5)
    search.fit(x_train, y_train, batch_size=101, epochs=500,
            validation_data=(x_valid, y_valid),
            callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)])
    
    matrix.append(str(i+1)+'_out')
    matrix.append(search.best_params_)
    matrix.append(search.best_score_)
    final_model = search.best_estimator_.model
    final_model.save('Ang_Accs_Inputs_Forces'+str(i+1)+'_out'+'.h5')

matrix = np.array(matrix)
 
if hostname=='PC64926':
    outfolder=r'D:\JuanCordero\NN_GRF_M_prediction_running\Grid_Search\\';
else:
    outfolder=r'C:\Users\juan_\Desktop\Tesis\Tesis_2\GridSearch\\';

with open(outfolder + 'Ang_Accs_Inputs_Forces.csv', 'x', newline='') as f:
    writer = csv.writer(f)      
    writer.writerows(zip(matrix))

