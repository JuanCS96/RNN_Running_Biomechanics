import os
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def rmse(y_true, y_pred):
    return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred-y_true), axis=-1))

os.chdir(r'C:\Users\....') # Type a valid path.

ann = "LSTM_ForceFrontalX.h5" # Load the appropiate RNN.
model = tf.keras.models.load_model(ann, custom_objects={'rmse': rmse})
    
file=pd.read_csv('INPUT_DATA_FILE.csv') # Type the correct file name.
hip_flex = file['hip_flexion_r']
knee_flex = file['knee_angle_r']
ankle_flex = file['ankle_angle_r']
hip_abd = file['hip_adduction_r']
hip_rot = file['hip_rotation_r']

force_vertical_max = 30.7338044862682
force_frontal_max = 3.962696769430389
force_lateral_max = 1.2553436247876946
hip_flex_moment_max = 0.9838254871914542
knee_flex_moment_max = 2.8792291654399333
ankle_flex_moment_max = 0.2369823246553613
hip_rot_moment_max = 0.7526198236525204
hip_abd_moment_max = 1.4531256188131314

x_data_full_2d=[]
y_data_full_2d=[]
for j in range(0, len(hip_flex)):
    x_data_full_2d.append([hip_flex[j], knee_flex[j], ankle_flex[j], hip_rot[j], hip_abd[j]])

for i in range(0, int(len(hip_flex)/101)):
    x_pred = np.array(x_data_full_2d[101*i:101*i+101]).reshape((101, 1, 5))
    y_pred=model.predict(x_pred, verbose=0)*force_frontal_max # This maximum value must change depending on the kinetic variable.

    #plt.plot(y_pred.squeeze())
#plt.show()
