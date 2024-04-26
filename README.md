# Comparison of Dynamic Prediction Data in Running: Skeletal Model vs. Artificial Neural Network-Based Approach 

<p align="center">

Recurrent Neural Networks to predict gorund reaction forces and lower limb joint moments from kinematic data in running.

## Requirements

- Python version 3.11.0.
- Tensorflow version 2.15.0 or higher.
- Numpy version 1.16.2 or higher.
- Pandas version 2.2.0 or higher.

---

## Instructions

- Prepare the input data. Input data are compose for the hip, knee and ankle flexion and hip abduction and rotation joint angles. Input data should be time normalized (101 data points) and scaled about the maximum joint angle value.
- Input data can be storage in a .csv file with one kinematic input variable per column. 
- Load the appropiate RNN to predict the desired kinetic variable.

<br>

See the example code to perform predictions.
