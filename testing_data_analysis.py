import tensorflow as tf
import numpy as np
import csv
import matplotlib.pyplot as plt
import time
import wandb
import os
import time
import pandas as pd
import argparse
import math

time_step = 5
num_data_type = 5
num_one_joint_data = time_step * (num_data_type-1)
num_joint = 6
num_input = num_one_joint_data*num_joint + time_step # joint data + ee_acc data
num_output = 2

tf.reset_default_graph()
sess = tf.Session()

new_saver = tf.train.import_meta_graph('model/model_test_robot12_input5_v5.ckpt.meta')
new_saver.restore(sess, 'model/model_test_robot12_input5_v5.ckpt')

graph = tf.get_default_graph()
x = graph.get_tensor_by_name("m1/input:0")
y = graph.get_tensor_by_name("m1/output:0")
keep_prob = graph.get_tensor_by_name("m1/keep_prob:0")
is_train = graph.get_tensor_by_name("m1/is_train:0")
#print([n.name for n in tf.get_default_graph().as_graph_def().node])
hypothesis = graph.get_tensor_by_name("m1/ConcatenateNet/hypothesis:0")
# Save Model
# saver = tf.train.Saver()
# saver.save(sess,'model/model.ckpt')


# Robot 3 Test Evaluation
TestData = pd.read_csv('../data/TestingDataNocut_robot3_input5_1000hz.csv').as_matrix().astype('float32')
X_Test = TestData[:,0:num_input]
Y_Test = TestData[:,-num_output:]
JTS = TestData[:,num_input]
DOB = TestData[:,num_input+1]
hypo  =  sess.run(hypothesis, feed_dict={x: X_Test, keep_prob: 1.0, is_train:False})
t = np.arange(0,0.001*len(JTS),0.001)

collision_pre = 0
collision_cnt = 0
collision_time = 0
detection_time_NN = []
detection_time_JTS = []
detection_time_DoB = []
collision_status = False
NN_detection = False
JTS_detection = False
DoB_detection = False
collision_fail_cnt_NN = 0
collision_fail_cnt_JTS = 0
collision_fail_cnt_DoB = 0

for i in range(len(JTS)):
    if (Y_Test[i,0] == 1 and collision_pre == 0):
        collision_cnt = collision_cnt +1
        collision_time = t[i]
        collision_status = True
        NN_detection = False
        JTS_detection = False
        DoB_detection = False
    
    if (collision_status == True and NN_detection == False):
        if(hypo[i,0] > 0.8):
            NN_detection = True
            detection_time_NN.append(t[i] - collision_time)

    if (collision_status == True and JTS_detection == False):
        if(JTS[i] == 1):
            JTS_detection = True
            detection_time_JTS.append(t[i] - collision_time)
    
    if (collision_status == True and DoB_detection == False):
        if(DOB[i] == 1):
            DoB_detection = True
            detection_time_DoB.append(t[i] - collision_time)

    if (Y_Test[i,0] == 0 and collision_pre == 1):
        collision_status = False
        if(NN_detection == False):
            detection_time_NN.append(0.0)
            collision_fail_cnt_NN = collision_fail_cnt_NN+1
        if(JTS_detection == False):
            detection_time_JTS.append(0.0)
            collision_fail_cnt_JTS = collision_fail_cnt_JTS+1
        if(DoB_detection == False):
            detection_time_DoB.append(0.0)
            collision_fail_cnt_DoB = collision_fail_cnt_DoB+1
    collision_pre = Y_Test[i,0]

print('Total collision (robot3): ', collision_cnt)
print('JTS Failure (robot3): ', collision_fail_cnt_JTS)
print('NN Failure (robot3): ', collision_fail_cnt_NN)
print('DOB Failure (robot3): ', collision_fail_cnt_DoB)
print('JTS Detection Time (robot3): ', sum(detection_time_JTS)/(collision_cnt - collision_fail_cnt_JTS))
print('NN Detection Time (robot3): ', sum(detection_time_NN)/(collision_cnt - collision_fail_cnt_NN))
print('DOB Detection Time (robot3): ', sum(detection_time_DoB)/(collision_cnt - collision_fail_cnt_DoB))


# Free motion Evaluation
TestDataFree = pd.read_csv('../data/TestingDataFree_robot3_input5_1000hz.csv').as_matrix().astype('float32')
X_TestFree = TestDataFree[:,0:num_input]
Y_TestFree = TestDataFree[:,-num_output:]
hypofree  =  sess.run(hypothesis, feed_dict={x: X_TestFree, keep_prob: 1.0, is_train:False})
false_positive_local_arr = np.zeros((len(Y_TestFree),1))
for j in range(len(Y_TestFree)):
    false_positive_local_arr[j] = hypofree[j,0] > 0.8  and np.equal(np.argmax(Y_TestFree[j,:]), 1)
print('False Positive (robot3): ', sum(false_positive_local_arr))
print('Total Num (robot3): ', len(Y_TestFree))


###################################################################################################


# Robot 3 Test Evaluation
TestData = pd.read_csv('../data/TestingDataNocut_robot1_input5_1000hz.csv').as_matrix().astype('float32')
X_Test = TestData[:,0:num_input]
Y_Test = TestData[:,-num_output:]
JTS = TestData[:,num_input]
DOB = TestData[:,num_input+1]
hypo  =  sess.run(hypothesis, feed_dict={x: X_Test, keep_prob: 1.0, is_train:False})
t = np.arange(0,0.001*len(JTS),0.001)

collision_pre = 0
collision_cnt = 0
collision_time = 0
detection_time_NN = []
detection_time_JTS = []
detection_time_DoB = []
collision_status = False
NN_detection = False
JTS_detection = False
DoB_detection = False
collision_fail_cnt_NN = 0
collision_fail_cnt_JTS = 0
collision_fail_cnt_DoB = 0

for i in range(len(JTS)):
    if (Y_Test[i,0] == 1 and collision_pre == 0):
        collision_cnt = collision_cnt +1
        collision_time = t[i]
        collision_status = True
        NN_detection = False
        JTS_detection = False
        DoB_detection = False
    
    if (collision_status == True and NN_detection == False):
        if(hypo[i,0] > 0.8):
            NN_detection = True
            detection_time_NN.append(t[i] - collision_time)

    if (collision_status == True and JTS_detection == False):
        if(JTS[i] == 1):
            JTS_detection = True
            detection_time_JTS.append(t[i] - collision_time)
    
    if (collision_status == True and DoB_detection == False):
        if(DOB[i] == 1):
            DoB_detection = True
            detection_time_DoB.append(t[i] - collision_time)

    if (Y_Test[i,0] == 0 and collision_pre == 1):
        collision_status = False
        if(NN_detection == False):
            detection_time_NN.append(0.0)
            collision_fail_cnt_NN = collision_fail_cnt_NN+1
        if(JTS_detection == False):
            detection_time_JTS.append(0.0)
            collision_fail_cnt_JTS = collision_fail_cnt_JTS+1
        if(DoB_detection == False):
            detection_time_DoB.append(0.0)
            collision_fail_cnt_DoB = collision_fail_cnt_DoB+1
    collision_pre = Y_Test[i,0]

print('Total collision (robot1): ', collision_cnt)
print('JTS Failure (robot1): ', collision_fail_cnt_JTS)
print('NN Failure (robot1): ', collision_fail_cnt_NN)
print('DOB Failure (robot1): ', collision_fail_cnt_DoB)
print('JTS Detection Time (robot1): ', sum(detection_time_JTS)/(collision_cnt - collision_fail_cnt_JTS))
print('NN Detection Time (robot1): ', sum(detection_time_NN)/(collision_cnt - collision_fail_cnt_NN))
print('DOB Detection Time (robot1): ', sum(detection_time_DoB)/(collision_cnt - collision_fail_cnt_DoB))


# Free motion Evaluation
TestDataFree = pd.read_csv('../data/TestingDataFree_robot1_input5_1000hz.csv').as_matrix().astype('float32')
X_TestFree = TestDataFree[:,0:num_input]
Y_TestFree = TestDataFree[:,-num_output:]
hypofree  =  sess.run(hypothesis, feed_dict={x: X_TestFree, keep_prob: 1.0, is_train:False})
false_positive_local_arr = np.zeros((len(Y_TestFree),1))
for j in range(len(Y_TestFree)):
    false_positive_local_arr[j] = hypofree[j,0] > 0.8  and np.equal(np.argmax(Y_TestFree[j,:]), 1)
print('False Positive (robot1): ', sum(false_positive_local_arr))
print('Total Num (robot1): ', len(Y_TestFree))