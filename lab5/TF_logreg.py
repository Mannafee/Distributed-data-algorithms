
import tensorflow as tf
import matplotlib.pyplot as plt   

#from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import train_test_split
from numpy.random import RandomState
from sklearn.datasets import fetch_olivetti_faces
from sklearn.preprocessing import OneHotEncoder





rng = RandomState(0)
learning_rate,epochs = 0.03, 100
epoch_count=np.arange(0,epochs)


dataset = fetch_olivetti_faces(shuffle=True, random_state=rng)
#faces = dataset.data
x = dataset.data
y = dataset.target

y = y.reshape(-1,1)
ybefore = np.copy(y)
oneHot = OneHotEncoder()
oneHot.fit(y) 
y = oneHot.transform(y).toarray() 


train_X, test_X, train_Y, test_Y = train_test_split(x, y, train_size=0.9)
m= train_X.shape[0] #number of training examples
n= train_X.shape[1] #features
z=train_Y.shape[1]
train_Y = train_Y.reshape(m,-1)





# There are n columns in the feature matrix 
 
X = tf.placeholder(tf.float32, [None, n]) 
Y = tf.placeholder(tf.float32, [None, z])  
W = tf.Variable(tf.zeros([n,z]), name="w")
b = tf.Variable(tf.zeros([1,z]), name="b")


# Hypothesis 
Y_hat = tf.nn.softmax(tf.matmul(X, W)+b)
  
#Cross Entropy Cost Function 
cost = -tf.reduce_mean(Y*tf.log(Y_hat))
  
# Gradient Descent Optimizer 
opt_sgd = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
#Adamoptimizer 
opt_adam=tf.train.AdamOptimizer(learning_rate).minimize(cost)
#RMSPropoptimizer
opt_prop=tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

  
# Global Variables Initializer 
init = tf.global_variables_initializer() 

correct_prediction = tf.equal(tf.argmax(Y_hat, 1), 
                                          tf.argmax(Y, 1)) 
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 
                                                 tf.float32)) 
          



# Starting the Tensorflow Session 
with tf.Session() as sess: 
      
    # Initializing the Variables 
    sess.run(init) 
      
    # Lists for storing the changing Cost and Accuracy in every Epoch 
    cost_history, train_accuracy,test_accuracy = [],[],[] 
      
    # Iterating through all the epochs training data
    for epoch in range(epochs): 
        cost_per_epoch = 0
#        for i in range(batch_size):
        # Running the Optimizer 
        sess.run(opt_prop, feed_dict = {X : train_X, Y : train_Y}) 
              
            # Calculating cost on current Epoch 
        c = sess.run(cost, feed_dict = {X : train_X, Y : train_Y}) 
          
        
        
        # Storing Cost and Accuracy to the history 
        cost_history.append(c)
        train_accuracy.append(accuracy.eval({X : train_X, Y : train_Y}) * 100) 
        test_accuracy.append(accuracy.eval({X : test_X, Y : test_Y}) * 100)
#        print("correct prediction ", sess.run(correct_prediction, feed_dict = {X:train_X,Y:train_Y}))
        # Displaying result on current Epoch 
        if epoch % 100 == 0 and epoch != 0: 
            print("Epoch " + str(epoch) + " Cost: "
                            + str(cost_history[-1])) 
      
#    W = sess.run(W) # Optimized Weight 
#    b= sess.run(b)   # Optimized Bias 
    plt.plot(epoch_count, train_accuracy, label='train accuracy ')
    plt.plot(epoch_count, test_accuracy, label='test accuracy')
    plt.legend()
    plt.show()
    
    plt.plot(epoch_count, cost_history, label='Loss graph of train data')
    plt.legend()
    plt.show()
  
    print("\nTraining Accuracy:", train_accuracy[-1], "%") 
    print("\nTesting Accuracy:", test_accuracy[-1],"%") 