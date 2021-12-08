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
import glob
import socket



hostname=socket.gethostname();
if hostname=='PC64926':
    os.chdir(r'D:\JuanCordero\NN_GRF_M_prediction_running')
else:
    os.chdir(r'C:\Users\juan_\Desktop\Tesis\Tesis_2\Tesis_3\Moments')

list = []
for o in glob.glob('*.xlsx'):
    list.append(o)

matrix = []
count = 1
for i in list:
    filename=i
    file=pd.read_excel(filename, sheet_name='Sheet1')

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
    #grf_x=file['GRF_x_15']
    #grf_y=file['GRF_y_15']
    #grf_z=file['GRF_z_15']
    hip_flex_m=file['Hip_flex_m']
    hip_abd_m=file['Hip_abd_m']
    hip_rot_m=file['Hip_rot_m']
    knee_flex_m=file['Knee_flex_m']
    ank_flex_m=file['Ank_flex_m']

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
    #grf_x=grf_x.tolist()
    #grf_y=grf_y.tolist()
    #grf_z=grf_z.tolist()
    hip_flex_m=hip_flex_m.tolist()
    hip_abd_m=hip_abd_m.tolist()
    hip_rot_m=hip_rot_m.tolist()
    knee_flex_m=knee_flex_m.tolist()
    ank_flex_m=ank_flex_m.tolist()


    x_data_full=[]
    y_data_full=[]
    for j in range(0, len(ankle_x)):
            x_data_full.append([ankle_x[j], ankle_y[j], ankle_z[j], knee_x[j], knee_y[j], knee_z[j], hip_x[j], hip_y[j], hip_z[j],
                                ankle_angular_vel_x[j], ankle_angular_vel_y[j], ankle_angular_vel_z[j],
                                ankle_angular_acc_x[j], ankle_angular_acc_y[j], ankle_angular_acc_z[j],
                                knee_angular_vel_x[j], knee_angular_vel_y[j], knee_angular_vel_z[j],
                                knee_angular_acc_x[j], knee_angular_acc_y[j], knee_angular_acc_z[j], 
                                hip_angular_vel_x[j], hip_angular_vel_y[j], hip_angular_vel_z[j],
                                hip_angular_acc_x[j], hip_angular_acc_y[j], hip_angular_acc_z[j],
                                pelvis_lineal_vel_x[j], pelvis_lineal_vel_y[j], pelvis_lineal_vel_z[j],
                                pelvis_lineal_acc_x[j], pelvis_lineal_acc_y[j], pelvis_lineal_acc_z[j], 
                                pelvis_angular_vel_x[j], pelvis_angular_vel_y[j], pelvis_angular_vel_z[j],
                                pelvis_angular_acc_x[j], pelvis_angular_acc_y[j], pelvis_angular_acc_z[j]])
            y_data_full.append([hip_flex_m[j], hip_abd_m[j], hip_rot_m[j], knee_flex_m[j], ank_flex_m[j]])


    x_train, x_valid = train_test_split(x_data_full, test_size=0.2, random_state=42)
    y_train, y_valid = train_test_split(y_data_full, test_size=0.2, random_state=42)


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
        model.add(keras.layers.Dense(5))
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
    
    matrix.append(str(count)+'_out')
    matrix.append(search.best_params_)
    matrix.append(search.best_score_)
    final_model = search.best_estimator_.model
    final_model.save('Ang_Vels_Accs_Inputs_Moments'+str(count)+'_out'+'.h5')
    count = count+1

matrix = np.array(matrix)

if hostname=='PC64926':
    outfolder=r'D:\JuanCordero\NN_GRF_M_prediction_running\Grid_Search\\';
else:
    outfolder=r'C:\Users\juan_\Desktop\Tesis\Tesis_2\Tesis_3\Moments\\';

with open(outfolder + 'Ang_Vels_Accs_Inputs_Moments.csv', 'x', newline='') as f:
    writer = csv.writer(f)      
    writer.writerows(zip(matrix))