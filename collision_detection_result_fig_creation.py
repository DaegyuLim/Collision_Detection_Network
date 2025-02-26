import tensorflow as tf
import numpy as np
import csv
import matplotlib.pyplot as plt
import argparse

# parameters
parser = argparse.ArgumentParser()
parser.add_argument('--data_type', type=str, default="collision")
args = parser.parse_args()

if args.data_type == "collision" :
    total_batch = 559
    file_name = '../data/TestSet/TestingDivide/collision_robot3_input5/Testing_data_robot3_'
elif args.data_type == "free":
    total_batch = 309
    file_name = '../data/TestSet/TestingDivide/Testing_data_free_'

time_step = 5
num_data_type = 5
num_one_joint_data = time_step * (num_data_type-1)
num_joint = 6
num_input = num_one_joint_data*num_joint  +  time_step # joint data + ee_acc data
num_output = 2
false_negative = 0.0
false_positive = 0.0

tf.reset_default_graph()
sess = tf.Session()

new_saver = tf.train.import_meta_graph('model/model_test_robot12_input5_v1.ckpt.meta')
new_saver.restore(sess, 'model/model_test_robot12_input5_v1.ckpt')

graph = tf.get_default_graph()
x = graph.get_tensor_by_name("m1/input:0")
y = graph.get_tensor_by_name("m1/output:0")
keep_prob = graph.get_tensor_by_name("m1/keep_prob:0")
is_train = graph.get_tensor_by_name("m1/is_train:0")
#print([n.name for n in tf.get_default_graph().as_graph_def().node])
hypothesis = graph.get_tensor_by_name("m1/ConcatenateNet/hypothesis:0")

accuracy_all = 0.0

#tensor = [n.name for n in tf.get_default_graph().as_graph_def().node]
# f = open("name.txt", 'w')
# for n in tensor:
#     f.write(n)
# f.close()

for i in range(total_batch): 
    path = file_name + str(i+1) + '.csv' # Testing_data_free_
    # raw data
    f = open(path, 'r', encoding='utf-8')
    rdr = csv.reader(f)
    t = []
    x_data_raw = []
    y_data_raw = []
    JTS = []
    DOB = []

    for line in rdr:
        line = [float(i) for i in line]
        x_data_raw.append(line[0:num_input])
        y_data_raw.append(line[-num_output:])
        JTS.append(line[num_input])
        DOB.append(line[num_input+1])
    t = range(len(x_data_raw))
    t = np.reshape(t,(-1,1))
    x_data_raw = np.reshape(x_data_raw, (-1, num_input))
    y_data_raw = np.reshape(y_data_raw, (-1, num_output))

    hypo = sess.run(hypothesis, feed_dict={x: x_data_raw, keep_prob: 1.0, is_train:False})

    prediction = np.argmax(hypo, 1)
    correct_prediction = np.equal(prediction, np.argmax(y_data_raw, 1))
    accuracy = np.mean(correct_prediction)
    accuracy_all = accuracy_all + accuracy


    print("Accuracy : %f" % accuracy)

    false_positive_local_arr = np.zeros_like(t)
    false_negative_local_arr = np.zeros_like(t)

    for j in range(len(x_data_raw)):
        false_positive_local_arr[j] = np.equal(prediction[j], 0) and np.equal(np.argmax(y_data_raw[j,:]), 1)
        false_negative_local_arr[j] = np.equal(prediction[j], 1) and np.equal(np.argmax(y_data_raw[j,:]), 0)

    false_positive_local = 0.0
    false_negative_local = 0.0
    if (np.sum(np.equal(np.argmax(y_data_raw,1), 1)) != 0):
        false_positive_local = np.sum(false_positive_local_arr)/np.sum(np.equal(np.argmax(y_data_raw,1), 1))
    if (np.sum(np.equal(np.argmax(y_data_raw,1), 0)) != 0):
        false_negative_local = np.sum(false_negative_local_arr)/np.sum(np.equal(np.argmax(y_data_raw,1), 0))
    false_positive += np.mean(false_positive_local)/total_batch
    false_negative += np.mean(false_negative_local)/total_batch
    print("False Positive Local: ", false_positive_local)
    print("False Negative Local: ", false_negative_local)
    print("False Positive: ",false_positive)
    print("False Negative: ",false_negative)

    if(i < total_batch):
        plt.plot(t,y_data_raw[:,0], color='r', marker="o", label='real')
        plt.plot(t,hypo[:,0], color='b',marker="x", label='prediction')
        plt.plot(t,JTS[:], color='k', marker="x", label='jts')
        plt.plot(t,DOB[:], color='y',marker="x", label='dob')
        plt.xlabel('time')
        plt.ylabel('Collision Probability')
        plt.legend()
        plt.ylim(0,1)
        plt.savefig('../FCModulize/result/test/Figure_' + str(i)+'.png')
        plt.clf()
        #plt.show()
        fileName = "../FCModulize/result/test/Result"+str(i)+".txt"
        savefile = open(fileName, 'w')
        np.savetxt(savefile, hypo[:,0])
        savefile.close()

accuracy_all = accuracy_all/total_batch
print(accuracy_all)