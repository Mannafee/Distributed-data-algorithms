import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
#from sklearn.utils import shuffle
import matplotlib.pyplot as plt

df=pd.read_csv("auto-mpg.data",header=None,sep=r"\s+")


#preprocessing the data
df=df.convert_objects(convert_numeric=True)
df=df.dropna()

y= np.array(df[0],dtype='float32')
x=np.array(df.loc[:,1:6],dtype='float32')


x = preprocessing.normalize(x)



#rng = np.random
learning_rate = 0.001
training_epochs = 100
epochs=np.arange(0,training_epochs)
training_cost=[]


train_X, test_X, train_Y, test_Y = train_test_split(x, y, train_size=0.9)
m= train_X.shape[0] #number of training examples
n= train_X.shape[1] #features
train_Y = train_Y.reshape(m,-1)

## Model linear regression y = Wx + b
X = tf.placeholder(tf.float32, [None,n])
Y= tf.placeholder(tf.float32, [None,1])
#
W = tf.Variable(tf.random_normal([1,n]), name="w")
b = tf.Variable(tf.constant(0.1), name="b")

pred = tf.add(tf.multiply(X,W),b)

# Mean squared error
cost_mse = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n)
cost_mae= tf.reduce_mean(tf.abs(Y - pred))
cost_rmse=tf.reduce_mean(tf.sqrt(tf.square(Y - pred)))

#
#
## Training using Gradient Descent to minimize cost
with tf.name_scope("train") as scope:
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_rmse)

init = tf.global_variables_initializer()


with tf.Session() as sess:

    # Run the initializer
    
    sess.run(init)

#     Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            x = x.reshape(-1,n)
            y = y.reshape(-1,1)
            sess.run(optimizer, feed_dict={X: x, Y: y})
        print('training data:number of epoch-->',epoch)
        training_cost.append(sess.run(cost_rmse, feed_dict={X: train_X, Y: train_Y}))
#    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')
    plt.plot(epochs, training_cost, label='Cost_RMSE')
#    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')

    plt.legend()
    plt.show()
