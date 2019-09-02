import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#from sklearn.utils import shuffle


rng = np.random

learning_rate = 0.1
training_epochs = 100


x =np.random.uniform(1,100,1000)


mu, sigma = 0, 50
noise = np.random.normal(mu, sigma, [1000,])
y = 0.5 * x + 2 + noise



train_X, test_X, train_Y, test_Y = train_test_split(x, y, train_size=0.9)
n_samples = train_X.shape[0]



# tf Graph Input
X = tf.placeholder("float")
Y = tf.placeholder("float")

#Set model weights
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")

# Construct a linear model
pred = tf.add(tf.multiply(X, W), b)

# Mean squared error
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)

# Gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize the variables 
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
#    test_accuracy=[]

    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})
        print('training data:number of epoch-->',epoch)
#        test_accuracy.append(accuracy.eval({X : test_X, Y : test_Y}) * 100)   

    
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
#    print("\nTesting Accuracy:", test_accuracy[-1],"%") 
#    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')

    plt.legend()
    plt.show()