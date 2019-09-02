import tensorflow as tf
import pickle
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

cifar10_dataset_folder_path = 'cifar-10-batches-py'
cost_history=[]
batch_cost_history=[]
final_cost_history=[]
accuracy_history=[]
batch_accuracy_history=[]
final_accuracy_history=[]

def variable_summaries(var,name):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries_'+name):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean_'+name, mean)
    tf.summary.scalar(name+'_value',var)
    tf.summary.histogram('histogram_'+name, var)
    
def load_label_names():
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def load_cfar10_batch(cifar10_dataset_folder_path, batch_id):
    with open(cifar10_dataset_folder_path + '/data_batch_' + str(batch_id), mode='rb') as file:
        # note the encoding type is 'latin1'
        batch = pickle.load(file, encoding='latin1')
        
    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']
    return features, labels

def normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x-min_val) / (max_val-min_val)
    return x

def one_hot_encode(x):
    encoded = np.zeros((len(x), 10))
    for idx, val in enumerate(x):
        encoded[idx][val] = 1
    return encoded

def data_aug(features,labels,augment_size=5000): 
    image_generator = ImageDataGenerator(rotation_range=90,horizontal_flip=True,vertical_flip=True)
    train_size=features.shape[0]
#   get transformed images
    randidx = np.random.randint(train_size, size=augment_size)

    labels=np.array(labels)
    x_augmented = features[randidx].copy()
    y_augmented = labels[randidx].copy()
    x_augmented = image_generator.flow(x_augmented, np.zeros(augment_size),
                                    batch_size=augment_size, shuffle=False).next()[0]
#   append augmented data to trainset
    trainx = np.concatenate((features, x_augmented))
    trainy= np.concatenate((labels, y_augmented)) 
    return trainx,trainy

def _preprocess_and_save(normalize, one_hot_encode,data_aug,features, labels, filename):
    features,labels=data_aug(features,labels)
    features = normalize(features)
    labels = one_hot_encode(labels)

    pickle.dump((features, labels), open(filename, 'wb'))
    
def minibatchoptimization(cifar10_dataset_folder_path, normalize, one_hot_encode,data_aug):
    n_batches = 5
    valid_features = []
    valid_labels = []

    for batch_i in range(1, n_batches + 1):
        features, labels = load_cfar10_batch(cifar10_dataset_folder_path, batch_i)
        
        # find index to be the point as validation data in the whole dataset of the batch (10%)
        index_of_validation = int(len(features) * 0.1)

        # preprocess the 90% of the whole dataset of the batch
        # - normalize the features
        # - one_hot_encode the lables
        # - save in a new file named, "preprocess_batch_" + batch_number
        # - each file for each batch
        _preprocess_and_save(normalize, one_hot_encode,data_aug,
                             features[:-index_of_validation], labels[:-index_of_validation], 
                             'preprocess_batch_' + str(batch_i) + '.p')

        # unlike the training dataset, validation dataset will be added through all batch dataset
        # - take 10% of the whold dataset of the batch
        # - add them into a list of
        #   - valid_features
        #   - valid_labels
        valid_features.extend(features[-index_of_validation:])
        valid_labels.extend(labels[-index_of_validation:])

    # preprocess the all stacked validation dataset
    _preprocess_and_save(normalize, one_hot_encode,data_aug,
                         np.array(valid_features), np.array(valid_labels),
                         'preprocess_validation.p')

    # load the test dataset
    with open(cifar10_dataset_folder_path + '/test_batch', mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')

    # preprocess the testing data
    test_features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    test_labels = batch['labels']

    # Preprocess and Save all testing data
    _preprocess_and_save(normalize, one_hot_encode,data_aug,
                         np.array(test_features), np.array(test_labels),
                         'preprocess_testing.p')
    
minibatchoptimization(cifar10_dataset_folder_path, normalize, one_hot_encode,data_aug)
valid_features, valid_labels = pickle.load(open('preprocess_validation.p', mode='rb'))
test_features, test_labels = pickle.load(open('preprocess_testing.p', mode='rb'))

tf.reset_default_graph()

# Inputs
x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name='input_x')
y =  tf.placeholder(tf.float32, shape=(None, 10), name='output_y')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

def conv_net(x, keep_prob):
    conv1_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 64], mean=0, stddev=0.08))

    
    conv1 = tf.nn.conv2d(x, conv1_filter, strides=[1,1,1,1], padding='SAME')
    conv1 = tf.nn.selu(conv1)
    conv1_pool_1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    conv1_bn = tf.layers.batch_normalization(conv1_pool_1)
    conv1_pool_2 = tf.nn.max_pool(conv1_bn, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')    
    conv1_bn = tf.layers.batch_normalization(conv1_pool_2)
    variable_summaries(conv1,'conv1')
    variable_summaries(conv1_pool_1,'conv1_pool')
#    variable_summaries(conv1,'conv1_bn')
    

    
    flat = tf.contrib.layers.flatten(conv1_bn)  
    full1 = tf.contrib.layers.fully_connected(inputs=flat, num_outputs=128, activation_fn=tf.nn.selu)
    full1 = tf.nn.dropout(full1, keep_prob)
    full1 = tf.layers.batch_normalization(full1)
    variable_summaries(full1,'full1')
    
    full2 = tf.contrib.layers.fully_connected(inputs=full1, num_outputs=256, activation_fn=tf.nn.selu)
    full2 = tf.nn.dropout(full2, keep_prob)
    full2 = tf.layers.batch_normalization(full2)
    variable_summaries(full2,'full2')
    


    out = tf.contrib.layers.fully_connected(inputs=full1, num_outputs=10, activation_fn=None)
    return out

epochs = 20
epoch_1=np.arange(0,epochs)
batch_size = 128
keep_probability= 0.7
learning_rate = 0.001

logits = conv_net(x, keep_prob)
model = tf.identity(logits, name='logits') # Name logits Tensor, so that can be loaded from disk after training

# Loss and Optimizer
my_normal_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels = y))
reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
reg_constant = 0.01  # Choose an appropriate one.
loss = my_normal_loss + reg_constant * sum(reg_losses)

tf.summary.scalar('loss', loss)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
tf.summary.scalar('accuracy', accuracy)


#tf.summary.histogram('accuary',accuracy)
#variable_summaries(loss,'loss')
#variable_summaries(accuracy, 'accuracy')
global_step = tf.Variable(0, dtype=tf.int32, trainable=False)

def train_neural_network(session, optimizer, keep_probability, feature_batch, label_batch):
    session.run(optimizer, 
                feed_dict={
                    x: feature_batch,
                    y: label_batch,
                    keep_prob: keep_probability
                })
def print_stats(session, feature_batch, label_batch, cost, accuracy):
    loss = session.run(cost, 
                    feed_dict={
                        x: feature_batch,
                        y: label_batch,
                        keep_prob: 1.
                    })
    cost_history.append(loss)
    
    valid_acc = session.run(accuracy,feed_dict={x: valid_features,y: valid_labels,keep_prob: 1.})
    accuracy_history.append(valid_acc)
    print('Loss: {:>10.4f} Validation Accuracy: {:.6f}'.format(loss, valid_acc))
    return cost_history,accuracy_history
    
    
def batch_features_labels(features, labels, batch_size):
    """
    Split features and labels into batches
    """
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        yield features[start:end], labels[start:end]

def load_preprocess_training_batch(batch_id, batch_size):
    """
    Load the Preprocessed Training data and return them in batches of <batch_size> or less
    """
    filename = 'preprocess_batch_' + str(batch_id) + '.p'
    features, labels = pickle.load(open(filename, mode='rb'))

    # Return the training data in batches of size <batch_size> or less
    return batch_features_labels(features, labels, batch_size)


save_model_path = r'C:\Users\manna\Documents\GitHub\CIFAR10-img-classification-tensorflow\graph_EX2'

print('Training...')
loss_over_time = np.zeros(epochs)
with tf.Session() as sess:
    merged_summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(r"C:\Users\manna\Documents\GitHub\CIFAR10-img-classification-tensorflow\graph_EX2", sess.graph)
    
    # Initializing the variables
    sess.run(tf.global_variables_initializer())
    
    # Training cycle
    for epoch in range(epochs):
        # Loop over all batches
        n_batches = 5
        for batch_i in range(1, n_batches + 1):
            for batch_features, batch_labels in load_preprocess_training_batch(batch_i, batch_size):
                train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
                
            print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
            cost_history,accuracy_history=print_stats(sess, batch_features, batch_labels, loss, accuracy)
            batch_cost_history.append(cost_history)
            batch_accuracy_history.append(accuracy_history)
        final_cost_history.append(np.mean(batch_cost_history))
        final_accuracy_history.append(np.mean(batch_accuracy_history))
    fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    fig.suptitle("loss vs epoch")
#    axs.plot(time_train,rmse_train,label='rmse_train')
    axs.plot(epoch_1,final_cost_history,label='loss',color='red')
    plt.legend()
    plt.show()
    
    fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    fig.suptitle("accuracy vs epoch")
#    axs.plot(time_train,rmse_train,label='rmse_train')
    axs.plot(epoch_1,final_accuracy_history,label='validation_accuracy',color='red')
    plt.legend()
    plt.show()
    print ("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_features,y:test_labels,keep_prob: 1.}))        
    # Save Model
    saver = tf.train.Saver()
    save_path = saver.save(sess, save_model_path)