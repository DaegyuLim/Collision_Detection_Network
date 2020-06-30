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
 
start_time = time.time()

# Parameters


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
parser = argparse.ArgumentParser()
parser.add_argument('--use_wandb', type=str2bool, default=True)
parser.add_argument('--use_gpu', type=str2bool, default=False)
parser.add_argument('--use_ee_acc_data', type=str2bool, default=True)
parser.add_argument('--learning_rate', type=float, default=1e-10)
parser.add_argument('--training_epoch', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--drop_out', type=float, default=1.0)
parser.add_argument('--regularization_factor', type=float, default=4e-6)
parser.add_argument('--hidden_neuron', type=int, default=16)
parser.add_argument('--cross_entropy_weight', type=float, default=0.01)
parser.add_argument('--input_type', type=str, default="input5")
parser.add_argument('--version', type=int, default=1)
parser.add_argument('--num_data_type', type=int, default=5)
args = parser.parse_args()

# Init wandb
wandb_use = args.use_wandb  
if wandb_use == True:
    wandb.init(project="Dusan_2nd_Project", name="finetuning_test_with_"+args.input_type+"_epoch_"+str(args.training_epoch)+"_ver"+str(args.version), tensorboard=False)

# Number of Input/Output Data
time_step = 5
num_data_type = args.num_data_type
num_one_joint_data = time_step * (num_data_type-1)
num_joint = 6
if args.use_ee_acc_data is False :
    num_input = num_one_joint_data*num_joint # joint data
    num_concatenate_node = 1*num_joint
else:
    num_input = num_one_joint_data*num_joint + 1* time_step # joint data + delta ee_acc data + delta current
    num_concatenate_node = 1*num_joint + 1
num_output = 2

# Hyper parameter Setting
learning_rate = args.learning_rate
training_epochs = args.training_epoch
batch_size = args.batch_size
drop_out = args.drop_out
regul_factor = args.regularization_factor
hidden_neurons = args.hidden_neuron
cross_entropy_weight = args.cross_entropy_weight
test_data_hz = 1000

# Tensorflow Setting
if args.use_gpu is False :
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    tf_config = tf.ConfigProto(
        allow_soft_placement=True,
        inter_op_parallelism_threads=16,
        intra_op_parallelism_threads=16, log_device_placement=False)
    # Prevent tensorflow from taking all the gpu memory
    tf_config.gpu_options.allow_growth = False
    sess = tf.Session(config=tf_config)
else :
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    sess = tf.Session()

# Neural Network
class Model:

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name, reuse = tf.AUTO_REUSE):
            self.X = tf.placeholder(tf.float32, shape=[None, num_input], name = "input")
            self.Y = tf.placeholder(tf.float32, shape=[None, num_output], name= "output")
            self.is_train = tf.placeholder(tf.bool, name = "is_train")
            self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
            self.cross_entropy_weight = tf.placeholder(tf.float32, name="cross_entropy_weight")
            self.hidden_layers = 0
            self.hidden_neurons = hidden_neurons

            # Joint Data Layers
            for i in range(6):
                with tf.variable_scope("Joint"+str(i)+"Net"):
                    W1 = tf.get_variable("W1", shape=[num_one_joint_data, self.hidden_neurons], initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(scale=regul_factor))
                    b1 = tf.Variable(tf.random_normal([self.hidden_neurons]))
                    L1 = tf.matmul(self.X[:, num_one_joint_data*i:num_one_joint_data*(i+1)], W1) +b1
                    L1 = tf.layers.batch_normalization(L1, training=self.is_train)
                    L1 = tf.nn.relu(L1)
                    L1 = tf.nn.dropout(L1, keep_prob=self.keep_prob)

                    W2 = tf.get_variable("W2", shape=[self.hidden_neurons, math.ceil(self.hidden_neurons)], initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(scale=regul_factor))
                    b2 = tf.Variable(tf.random_normal([math.ceil(self.hidden_neurons)]))
                    L2 = tf.matmul(L1, W2) +b2
                    L2 = tf.layers.batch_normalization(L2, training=self.is_train)
                    L2 = tf.nn.relu(L2)
                    L2 = tf.nn.dropout(L2, keep_prob=self.keep_prob)
                    self.hidden_layers += 1

                    W3 = tf.get_variable("W3", shape=[math.ceil(self.hidden_neurons), math.ceil(self.hidden_neurons)], initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(scale=regul_factor))
                    b3 = tf.Variable(tf.random_normal([math.ceil(self.hidden_neurons)]))
                    L3 = tf.matmul(L2, W3) +b3
                    L3 = tf.layers.batch_normalization(L3, training=self.is_train)
                    L3 = tf.nn.relu(L3)
                    L3 = tf.nn.dropout(L3, keep_prob=self.keep_prob)
                    self.hidden_layers += 1

                    W4 = tf.get_variable("W4", shape=[math.ceil(self.hidden_neurons), 1], initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(scale=regul_factor))
                    b4 = tf.Variable(tf.random_normal([1]))
                    L4 = tf.matmul(L3, W4) +b4
                    L4 = tf.layers.batch_normalization(L4, training=self.is_train)
                    L4 = tf.nn.relu(L4)
                    L4 = tf.nn.dropout(L4, keep_prob=self.keep_prob)
                    self.hidden_layers += 1

                    if(i == 0):
                        self.LConcat = L4
                    else:
                        self.LConcat = tf.concat([self.LConcat, L4],1)
                        
            if args.use_ee_acc_data is True :
                # End Effector Accerlation Data Layers
                W_ee1 = tf.get_variable("W_ee1", shape=[1*time_step, self.hidden_neurons], initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(regul_factor))
                b_ee1 = tf.Variable(tf.random_normal([self.hidden_neurons]))
                L_ee1 = tf.matmul(self.X[:, num_one_joint_data*6:num_one_joint_data*6+1*time_step], W_ee1) + b_ee1
                L_ee1 = tf.layers.batch_normalization(L_ee1, training=self.is_train)
                L_ee1 = tf.nn.relu(L_ee1)
                L_ee1 = tf.nn.dropout(L_ee1, keep_prob=self.keep_prob)

                W_ee2 = tf.get_variable("W_ee2", shape=[self.hidden_neurons, 1], initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(regul_factor))
                b_ee2 = tf.Variable(tf.random_normal([1]))
                L_ee2 = tf.matmul(L_ee1, W_ee2) +b_ee2
                L_ee2 = tf.layers.batch_normalization(L_ee2, training=self.is_train)
                L_ee2 = tf.nn.relu(L_ee2)
                L_ee2 = tf.nn.dropout(L_ee2, keep_prob=self.keep_prob)
                self.hidden_layers += 1
                self.LConcat = tf.concat([self.LConcat, L_ee2],1)

            with tf.variable_scope("ConcatenateNet"):
                W5 = tf.get_variable("W5", shape=[num_concatenate_node, math.ceil(self.hidden_neurons)], initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(scale=regul_factor))
                b5 = tf.Variable(tf.random_normal([math.ceil(self.hidden_neurons)]))
                L5 = tf.matmul(self.LConcat, W5) +b5
                L5 = tf.layers.batch_normalization(L5, training=self.is_train)
                L5 = tf.nn.relu(L5)
                L5 = tf.nn.dropout(L5, keep_prob=self.keep_prob)
                self.hidden_layers += 1

                W6 = tf.get_variable("W6", shape=[math.ceil(self.hidden_neurons), math.ceil(self.hidden_neurons)], initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(scale=regul_factor))
                b6 = tf.Variable(tf.random_normal([math.ceil(self.hidden_neurons)]))
                L6 = tf.matmul(L5, W6) +b6
                L6 = tf.layers.batch_normalization(L6, training=self.is_train)
                L6 = tf.nn.relu(L6)
                L6 = tf.nn.dropout(L6, keep_prob=self.keep_prob)
                self.hidden_layers += 1

                W7 = tf.get_variable("W7", shape=[math.ceil(self.hidden_neurons), num_output], initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(scale=regul_factor))
                b7 = tf.Variable(tf.random_normal([num_output]))
                self.logits = tf.matmul(L6, W7) +b7
                tf.identity(self.logits, "logits")
                self.hypothesis = tf.nn.softmax(self.logits)
                
                self.hypothesis = tf.identity(self.hypothesis, "hypothesis")

            # define cost/loss & optimizer
            self.l2_reg = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            # self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.Y))
            self.cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=self.Y, logits=self.logits, pos_weight=cross_entropy_weight))

            self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(self.update_ops):
                self.optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate).minimize(self.cost + self.l2_reg)
        
        self.prediction = tf.argmax(self.hypothesis, 1)
        self.correct_prediction = tf.equal(self.prediction, tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    def get_mean_error_hypothesis(self, x_test, y_test, keep_prop=1.0, is_train=False):
        return self.sess.run([self.accuracy,  self.l2_reg, self.cost, self.hypothesis], feed_dict={self.X: x_test, self.Y: y_test, self.keep_prob: keep_prop, self.is_train: is_train})

    def train(self, x_data, y_data, keep_prop=1.0, is_train=True):
        return self.sess.run([self.accuracy, self.l2_reg, self.cost, self.optimizer], feed_dict={
            self.X: x_data, self.Y: y_data, self.keep_prob: keep_prop, self.is_train: is_train})

    def get_hidden_number(self):
        return [self.hidden_layers, self.hidden_neurons]

m1 = Model(sess, "m1")
sess.run(tf.global_variables_initializer())
new_saver = tf.train.import_meta_graph('model/model_test_robot12_input5_v1.ckpt.meta')
new_saver.restore(sess, 'model/model_test_robot12_input5_v1.ckpt')

# graph = tf.get_default_graph()


# Log Configuration
if wandb_use == True:
    wandb.config.epoch = training_epochs
    wandb.config.batch_size = batch_size
    wandb.config.learning_rate = learning_rate
    wandb.config.drop_out = drop_out
    wandb.config.num_input = num_input
    wandb.config.num_output = num_output
    wandb.config.time_step = time_step
    wandb.config.hidden_layers, wandb.config.hidden_neurons = m1.get_hidden_number()
    wandb.config.L2_regularization = regul_factor 

# Parse .tfrecord Data
def parse_proto(example_proto):
  features = {
    'X': tf.FixedLenFeature((num_input,), tf.float32),
    'y': tf.FixedLenFeature((num_output,), tf.float32),
  }
  parsed_features = tf.parse_single_example(example_proto, features)
  return parsed_features['X'], parsed_features['y']

# Load Training Data with tf.data
TrainData = tf.data.TFRecordDataset(["../data/TrainingData_robot3_free_"+args.input_type+".tfrecord"])
TrainData = TrainData.shuffle(buffer_size=20*batch_size)
TrainData = TrainData.map(parse_proto)
TrainData = TrainData.batch(batch_size)
#TrainData = TrainData.prefetch(buffer_size=100)
Trainiterator = TrainData.make_initializable_iterator()
train_batch_x, train_batch_y = Trainiterator.get_next()

# Load Validation Data in Memory
ValidationData = pd.read_csv('../data/ValidationData_robot3_'+args.input_type+'.csv').as_matrix().astype('float64')
X_validation = ValidationData[:,0:num_input]
Y_validation = ValidationData[:,-num_output:]

print('size of train_batch_x: ', train_batch_x.shape)
print('size of train_batch_y: ', train_batch_y.shape)

# Train Model
train_acc = np.zeros(training_epochs)
validation_acc = np.zeros(training_epochs)
train_cost = np.zeros(training_epochs)
validation_cost = np.zeros(training_epochs)

# To Scale wandb Charts
if wandb_use == True:
    wandb_dict = dict()
    wandb_dict['Training Accuracy'] = 0.0
    wandb_dict['Validation Accuracy'] = 0.0
    wandb_dict['Training Cost'] = 1.5
    wandb_dict['Validation Cost'] = 1.5
    wandb.log(wandb_dict)

for epoch in range(training_epochs):
    accu_train = 0
    accu_val = 0
    reg_train = 0
    reg_val = 0
    cost_train = 0
    cost_val = 0
    train_batch_num = 0
    validation_batch_num = 0
    
    # Training Data
    sess.run(Trainiterator.initializer)
    while True:
        try:
            # read_start_time = time.time()
            x,y = sess.run([train_batch_x, train_batch_y])
            # read_finish_time = time.time()
            # print('read time : ', read_finish_time-read_start_time)
            if (x.shape[0]==batch_size):
                accu, reg_c, cost,_ = m1.train(x, y, drop_out)
                # print('train time : ', time.time()-read_finish_time)
                train_batch_num = train_batch_num + 1
                accu_train = ((train_batch_num-1)*accu_train + accu )/ train_batch_num
                reg_train = ((train_batch_num-1)*reg_train + reg_c )/ train_batch_num
                cost_train = ((train_batch_num-1)*cost_train + cost )/ train_batch_num
        except tf.errors.OutOfRangeError:
            break
    # Validation Evaluation
    accu_val, reg_val, cost_val, _ = m1.get_mean_error_hypothesis(X_validation, Y_validation)

    print('Epoch:', '%04d' % (epoch + 1))
    print('Train Accuracy =', '{:.9f}'.format(accu_train))
    print('Validation Accuracy =', '{:.9f}'.format(accu_val))
    print('Train Cost =', '{:.9f}'.format(cost_train), 'Train Regul =', '{:.9f}'.format(reg_train))
    print('Validation Cost =', '{:.9f}'.format(cost_val), 'Validation Regul =', '{:.9f}'.format(reg_val))

    train_acc[epoch] = accu_train
    validation_acc[epoch] = accu_val
    train_cost[epoch] = cost_train
    validation_cost[epoch] = cost_val

    # Log to wandb
    if wandb_use == True:
        wandb_dict = dict()
        wandb_dict['Training Accuracy'] = accu_train
        wandb_dict['Validation Accuracy'] = accu_val
        wandb_dict['Training Cost'] = cost_train
        wandb_dict['Training Reg'] = reg_train
        wandb_dict['Validation Cost'] = cost_val
        wandb_dict['Validation Reg'] = reg_val
        
        if epoch % 20 ==0:
            for var in tf.trainable_variables():
                wandb_dict[var.name] =sess.run(var)
        wandb.log(wandb_dict)

elapsed_time = time.time() - start_time
print(elapsed_time)
print('Learning Finished!')

# Save Model
# saver = tf.train.Saver()
# saver.save(sess,'model/model_FT_robot3_'+args.input_type+'_epoch_'+str(args.training_epoch)+'_v'+str(args.version)+'.ckpt')

# if wandb_use == True:
#     saver.save(sess, os.path.join(wandb.run.dir, 'model/model_FT_robot3_'+args.input_type+'_epoch_'+str(args.training_epoch)+'_v'+str(args.version)+'.ckpt'))
#     wandb.config.elapsed_time = elapsed_time

########################### robot3 Test Evaluation ##############################
TestData = pd.read_csv('../data/TestingDataNocut_robot3_'+args.input_type+'_1000hz.csv').as_matrix().astype('float32')
X_Test = TestData[:,0:num_input]
Y_Test = TestData[:,-num_output:]
JTS = TestData[:,num_input]
DOB = TestData[:,num_input+1]
accu_test, reg_test, cost_test, hypo  = m1.get_mean_error_hypothesis(X_Test, Y_Test)
# prediction = np.argmax(hypo, 1)
prediction = hypo[:, 0]
t = np.arange(0, len(prediction)/test_data_hz, 1/test_data_hz)
print('Test Accuracy(Robot3): ', accu_test)
print('Test Cost(Robot3): ', cost_test)

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

for i in range(len(prediction)):
    if (Y_Test[i, 0] == 1 and collision_pre == 0):
        collision_cnt = collision_cnt +1
        collision_time = t[i]
        collision_status = True
        NN_detection = False
        JTS_detection = False
        DoB_detection = False
    
    if (collision_status == True and NN_detection == False):
        if(prediction[i] >= 0.8):
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

    if (Y_Test[i, 0] == 0 and collision_pre == 1):
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
    collision_pre = Y_Test[i, 0]

print('Total collision(Robot3): ', collision_cnt)
print('JTS Failure(Robot3): ', collision_fail_cnt_JTS)
print('NN Failure(Robot3): ', collision_fail_cnt_NN)
print('DOB Failure(Robot3): ', collision_fail_cnt_DoB)
if((collision_cnt - collision_fail_cnt_JTS) != 0):
    print('JTS Detection Time(Robot3): ', sum(detection_time_JTS)/(collision_cnt - collision_fail_cnt_JTS))
if((collision_cnt - collision_fail_cnt_NN) != 0):
    print('NN Detection Time(Robot3): ', sum(detection_time_NN)/(collision_cnt - collision_fail_cnt_NN))
if((collision_cnt - collision_fail_cnt_DoB) != 0):
    print('DOB Detection Time(Robot3): ', sum(detection_time_DoB)/(collision_cnt - collision_fail_cnt_DoB))


# Robot3 Free motion Evaluation 
TestDataFree = pd.read_csv('../data/TestingDataFree_robot3_'+args.input_type+'_1000hz.csv').as_matrix().astype('float32')
X_TestFree = TestDataFree[:,0:num_input]
Y_TestFree = TestDataFree[:,-num_output:]
accu_test, reg_test, cost_test, hypo  = m1.get_mean_error_hypothesis(X_TestFree, Y_TestFree)
prediction = hypo[:, 0]
false_positive_local_arr = np.zeros((len(Y_TestFree),1))
for j in range(len(false_positive_local_arr)):
    false_positive_local_arr[j] = (prediction[j]>= 0.8) and np.equal(np.argmax(Y_TestFree[j,:]), 1)
print('False Positive (Robot3): ', sum(false_positive_local_arr))
print('Total Num (Robot3): ', len(Y_TestFree))




############################### robot1 Test Evaluation ######################
TestData = pd.read_csv('../data/TestingDataNocut_robot1_'+args.input_type+'_1000hz.csv').as_matrix().astype('float32')
X_Test = TestData[:,0:num_input]
Y_Test = TestData[:,-num_output:]
JTS = TestData[:,num_input]
DOB = TestData[:,num_input+1]
t = np.arange(0, len(prediction)/test_data_hz, 1/test_data_hz)
accu_test, reg_test, cost_test, hypo  = m1.get_mean_error_hypothesis(X_Test, Y_Test)
# prediction = np.argmax(hypo, 1)
prediction = hypo[:, 0]
print('Test Accuracy(Robot1): ', accu_test)
print('Test Cost(Robot1): ', cost_test)

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

for i in range(len(prediction)):
    if (Y_Test[i, 0] == 1 and collision_pre == 0):
        collision_cnt = collision_cnt +1
        collision_time = t[i]
        collision_status = True
        NN_detection = False
        JTS_detection = False
        DoB_detection = False
    
    if (collision_status == True and NN_detection == False):
        if(prediction[i] >= 0.8):
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

    if (Y_Test[i, 0] == 0 and collision_pre == 1):
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
    collision_pre = Y_Test[i, 0]

print('Total collision(Robot1): ', collision_cnt)
print('JTS Failure(Robot1): ', collision_fail_cnt_JTS)
print('NN Failure(Robot1): ', collision_fail_cnt_NN)
print('DOB Failure(Robot1): ', collision_fail_cnt_DoB)
if((collision_cnt - collision_fail_cnt_JTS) != 0):
    print('JTS Detection Time(Robot1): ', sum(detection_time_JTS)/(collision_cnt - collision_fail_cnt_JTS))
if((collision_cnt - collision_fail_cnt_NN) != 0):
    print('NN Detection Time(Robot1): ', sum(detection_time_NN)/(collision_cnt - collision_fail_cnt_NN))
if((collision_cnt - collision_fail_cnt_DoB) != 0):
    print('DOB Detection Time(Robot1): ', sum(detection_time_DoB)/(collision_cnt - collision_fail_cnt_DoB))


# Robot1 Free motion Evaluation 
TestDataFree = pd.read_csv('../data/TestingDataFree_robot1_'+args.input_type+'_1000hz.csv').as_matrix().astype('float32')
X_TestFree = TestDataFree[:,0:num_input]
Y_TestFree = TestDataFree[:,-num_output:]
accu_test, reg_test, cost_test, hypo  = m1.get_mean_error_hypothesis(X_TestFree, Y_TestFree)
prediction = hypo[:, 0]
false_positive_local_arr = np.zeros((len(Y_TestFree),1))
for j in range(len(false_positive_local_arr)):
    false_positive_local_arr[j] = (prediction[j]>= 0.8) and np.equal(np.argmax(Y_TestFree[j,:]), 1)
print('False Positive (Robot1): ', sum(false_positive_local_arr))
print('Total Num (Robot1): ', len(Y_TestFree))
