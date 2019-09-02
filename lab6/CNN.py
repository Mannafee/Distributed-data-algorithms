import tensorflow as tf
import pandas as pd
from scipy import stats
from sklearn.model_selection import LeaveOneOut
import glob
import numpy as np

import matplotlib.pyplot as plt






#*****importing all the data and merged into one dataframe except subject 108 and subject 109**********
path = r'E:\Documents\University of Hildesheim\Distributed data analytics\lab6\PAMAP2_Dataset\Protocol' # use your path
all_files = glob.glob(path + "/*.dat")
data = []
X=[]
for filename in all_files:
    df=pd.read_csv(filename,header=None,index_col=None,sep=r"\s+")
    df=df.convert_objects(convert_numeric=True)
    df=df.dropna()
    data.append(df)

#***************selecting specific ID******************    
act_id=[3,4,12,13]
data_four_act=[]
for x in range (len(data)):
    data_2=data[x]
    for i in range (len(act_id)):
#        print(i)
        data_four_act.append(data_2[data_2[1] ==act_id[i]])
    X.append(pd.concat(data_four_act, axis=0, ignore_index=True))    



# FUNCTION DECLARATION
def feature_normalize(dataset):
    mu = np.mean(dataset,axis = 0)
    sigma = np.std(dataset,axis = 0)
    return (dataset - mu)/sigma


#for TensorBoard visualization
def variable_summaries(var,name):
  with tf.name_scope('summaries_'+name):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean_'+name, mean)
    tf.summary.scalar(name+'_value',var)
    tf.summary.histogram('histogram_'+name, var)
    
def window(data, size):
    start = 0
    while start < len(data):
        yield start, start + size
        start += int(size / 2)
def slide_window(x_train,y_train,window_size):
    seg = np.zeros(((len(x_train)//(window_size//2))-1,window_size,3))
    labels = np.zeros(((len(y_train)//(window_size//2))-1))
    i_seg = 0
    i_label = 0
    
    for (start,end) in window(x_train,window_size):
#        print(start,end)
        if(len(x_train[start:end]) == window_size):
            m = stats.mode(y_train[start:end])
            seg[i_seg] = x_train[start:end]
            labels[i_label] = m[0]
            i_label+=1
            i_seg+=1
    return seg, labels

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)
    
def depth_conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool(x,kernel_size,stride_size):
  return tf.nn.max_pool(x, ksize=[1, 1, kernel_size, 1],
                        strides=[1, 1, stride_size, 1], padding='VALID')
  

    
    

##***********leave one out cross validation***********************

loo = LeaveOneOut()
for train_index, test_index in loo.split(X):
   train_X=[] 
   
   print("TRAIN:", train_index, "TEST:", test_index)
   for i in range (len(train_index)):
       train_2=[]
       train_2.append(X[train_index[i]])
   train_X=pd.concat(train_2, axis=0, ignore_index=True)
   trainx=(feature_normalize(train_X.iloc[:, np.r_[4:7]]))##*************3D column normalization****************
   test_X=X[test_index[0]]
   testx=(feature_normalize(test_X.iloc[:, np.r_[4:7]]))##*************3D column normalization****************
   trainy=train_X[1].values
   testy=test_X[1].values
   
   window_size=256
   print ("sliding window........")
   train_x, train_y = slide_window(trainx,trainy,window_size)
   test_x, test_y = slide_window(testx,testy,window_size)
   
   train_1 = pd.get_dummies(train_y)
   test = pd.get_dummies(test_y)
   train_1, test = train_1.align(test, join='inner', axis=1)
   train_y = np.asarray(train_1)
   test_y = np.asarray(test)
   
   input_height = 1
   window_size = window_size
   num_labels = 4  
   num_channels = 3 
   batch_size = 128
   stride_size = 2
   kernel_size_1 = 7
   kernel_size_2 = 3
   kernel_size_3 = 1
   depth_1 = 256
   depth_2 = 256
   depth_3 = 256
   num_hidden = 512 # neurons in the fully connected layer
   dropout_1 = tf.placeholder(tf.float32) 
   dropout_2 = tf.placeholder(tf.float32) 
   dropout_3 = tf.placeholder(tf.float32) 

   learning_rate = 0.0005
   training_epochs = 10
   total_batches = train_x.shape[0] // batch_size
   train_x = train_x.reshape(len(train_x),1, window_size,num_channels) # opportunity
   test_x = test_x.reshape(len(test_x),1, window_size,num_channels) # opportunity
   
   X = tf.placeholder(tf.float32, shape=[None,input_height,window_size,num_channels])
   Y = tf.placeholder(tf.float32, shape=[None,num_labels])
   
   # hidden layer 1
   W_conv1 = weight_variable([1, kernel_size_1, num_channels, depth_1])
   b_conv1 = bias_variable([depth_1])
   h_conv1 = tf.nn.relu(depth_conv2d(X, W_conv1) + b_conv1)
   # h_conv1 = tf.nn.dropout(tf.identity(h_conv1), dropout_1)
   h_conv1 = tf.nn.dropout(h_conv1, dropout_1)
   h_pool1 = max_pool(h_conv1,kernel_size_1,stride_size)
   
   # hidden layer 2
   W_conv2 = weight_variable([1, kernel_size_2, depth_1, depth_2])
   b_conv2 = bias_variable([depth_2])
   h_conv2 = tf.nn.relu(depth_conv2d(h_pool1, W_conv2) + b_conv2)
   h_conv2 = tf.nn.dropout(h_conv2, dropout_2)   
   h_pool2 = max_pool(h_conv2,kernel_size_2,stride_size)
   
   # fully connected layer

    #first we get the shape of the last layer and flatten it out
   shape = h_pool2.get_shape().as_list()
#   print ("shape's shape:", shape)

   W_fc1 = weight_variable([shape[1] * shape[2] * shape[3],num_hidden])
   b_fc1 = bias_variable([num_hidden])

   h_pool3_flat = tf.reshape(h_pool2, [-1, shape[1] * shape[2] * shape[3]])
    #print "c_flat shape =",h_pool3_flat.shape
   h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat,W_fc1) + b_fc1)
   h_fc1 = tf.nn.dropout(h_fc1, dropout_3)
   
    #readout layer.
   W_fc2 = weight_variable([num_hidden,num_labels])
   b_fc2 = bias_variable([num_labels])
   y_conv = tf.matmul(h_fc1,W_fc2) + b_fc2
    
    # COST FUNCTION
   loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=y_conv))
#   loss=tf.reduce_mean(tf.sqrt(tf.square(Y - y_conv)))
   optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(loss)
#   optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)
   correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(Y,1))
   accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
   loss_over_time = np.zeros(training_epochs)
   with tf.Session() as session:

    # merged_summary_op = tf.summary.merge_all()
    # summary_writer = tf.summary.FileWriter("./", session.graph)
    tf.initialize_all_variables().run()

    for epoch in range(training_epochs):

        cost_history = np.empty(shape=[0],dtype=float)
        for b in range(total_batches):
            offset = (b * batch_size) % (train_y.shape[0] - batch_size)
            batch_x = train_x[offset:(offset + batch_size), :, :, :]
            batch_y = train_y[offset:(offset + batch_size),:]
            
#            print ("batch_x shape =",batch_x.shape)
#            print ("batch_y shape =",batch_y.shape)
            # _, c, summary= session.run([optimizer, loss,merged_summary_op],feed_dict={X: batch_x, Y : batch_y, dropout_1: 1-0.1, dropout_2: 1-0.25, dropout_3: 1-0.5})
#             cost_history = np.append(cost_history,c)
#            print(optimizer,loss)
            _,c= session.run([optimizer, loss],feed_dict={X: batch_x, Y : batch_y, dropout_1: 1-0.1, dropout_2: 1-0.25, dropout_3: 1-0.5})
#            print(c)
            cost_history = np.append(cost_history,c)
            # summary_writer.add_summary(summary,global_step.eval(session=session))
        print ("Epoch: ",epoch," Training Loss: ",np.mean(cost_history)," Training Accuracy: ",session.run(accuracy, feed_dict={X: train_x, Y: train_y, dropout_1: 1-0.1, dropout_2: 1-0.25, dropout_3: 1-0.5}))
        loss_over_time[epoch] = np.mean(cost_history)
    
    print ("Testing Accuracy:", session.run(accuracy, feed_dict={X: test_x, Y: test_y, dropout_1: 1, dropout_2: 1, dropout_3: 1}))
   
   