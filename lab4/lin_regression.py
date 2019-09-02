import os
#import glob
from mpi4py import MPI
import numpy as np
import random
from random import shuffle
import math as Math
from sklearn.datasets import load_svmlight_file
import matplotlib.pyplot as plt
#import pdb
file_path=r"C:\Users\manna\Desktop\data"


epochs=100
epoch=np.arange(0,epochs)

rmse_train=[]
rmse_test=[]
weight_epoch=[]
#time=[28.712002599990228,26.835568099981174,14.847459999989951,12.351702599989949,15.926856200007023]
time_train=[]
time_test=[]



comm = MPI.COMM_WORLD
num_workers = comm.Get_size()
worker= comm.Get_rank()


def collect_data(chunks):           
    for i in range((len(chunks))-1):            
        #decoding sparse data
        
#        print(chunks[i])
        tup= load_svmlight_file(chunks[i])
        y=tup[1]
        x=tup[0].todense()
        if x.shape[1]<479:
            n,m = x.shape # for generality
            x0= np.zeros((n,1))
            x = np.hstack((x,x0))

    return x,y


def SGD(x,y):
    
    w=np.zeros(x.shape[1])
    l=0.0000000001 #learning rate

    for k in range(epochs):
        
        print('train','number of epoch is:',k)
        
        indexes = np.arange(0,x.shape[0])
        shuffle(indexes)
        for i in indexes:
                
            y_pred_train=w.dot((np.transpose(x[i])))
            d_m=-(y[i]-y_pred_train[0,0])*x[i]
            w=w-l*d_m            
        y_pred_train=w.dot(np.transpose(x))
        rmse_train.append(rmse(y,y_pred_train))
        weight_epoch.append(w)
        end_train = MPI.Wtime()
        time_train.append(end_train-start_train)        
#        print('epoch:',k,w)
    return weight_epoch


def rmse(y,y_pred):
    y_pred = np.squeeze(np.asarray(y_pred))
    rmse= Math.sqrt(np.sum(pow((y-y_pred),2)))/len(y)
#    print(rmse)
    return rmse
 


if worker==0:
    files=[]
    new_weight=[]
    for file in os.listdir(file_path):
        if file.endswith(".txt"):
            files.append(os.path.join(file_path, file))
    random.shuffle(files)
    train_data=int(len(files)*0.7)
    test_data=len(files)
    chunks=np.array_split(files[0:train_data],num_workers)
#    print('chunk',chunks)
else:
    chunks=None
    files=None
    new_weight=None

vA=comm.scatter(chunks,root=0)
vB=comm.bcast(new_weight,root=0)


start_train = MPI.Wtime() 
x,y=collect_data(vA)    
vB.append(SGD(x,y))



if worker==0:    
    average_weight=np.mean(vB)
    print("average weight of training data =",average_weight)
    
    
    start_test=MPI.Wtime() 
    x,y=collect_data(files[train_data:test_data])
    for k in range(epochs):
        print('test','number of epoch is:',k)
        
        y_pred_test=vB[0][k].dot(np.transpose(x))
        rmse_test.append(rmse(y,y_pred_test))
        end_test = MPI.Wtime()
        time_test.append(end_test-start_test)
#    print("Runtime", end-start) 
    
    fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    fig.suptitle("RMSE vs Time")
#    axs.plot(time_train,rmse_train,label='rmse_train')
    axs.plot(time_test,rmse_test,label='rmse_test',color='red')
    axs.set_ylabel('RMSE values')
    axs.set_xlabel('Time')
    plt.legend()
    plt.show()
   
    
    
    fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    fig.suptitle("RMSE vs Time")
    axs.plot(time_train,rmse_train,label='rmse_train')
#    axs.plot(time_test,rmse_test,label='rmse_test')
    axs.set_ylabel('RMSE values')
    axs.set_xlabel('Time')
    plt.legend()
    plt.show()
    
#    plt.plot(epoch,rmse_train, label = "line 1")
#    plt.plot(epoch,rmse_test, label = "line 1")



    
    
    
    




