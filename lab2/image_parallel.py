


from mpi4py import MPI
import cv2
#from matplotlib import pyplot as plt
import numpy as np
import collections


    
comm = MPI.COMM_WORLD
num_workers = comm.Get_size()
worker= comm.Get_rank()

def frequency(data):    
    global c
    c = collections.Counter(data)

if worker == 0:
     
     imgGray = cv2.imread('image.jpg',0)
     
     imgRGB=cv2.imread('image.jpg')
     imgRGB=cv2.cvtColor(imgRGB,cv2.COLOR_BGR2RGB)
     
     arr  = np.asarray(imgGray)
     flat = arr.reshape(np.prod(arr.shape[:2]),-1)
     scal_flat=np.zeros(len(flat))
     freq_size=np.amax(flat)
     freq=np.zeros(freq_size+1)
     for i in range (len(flat)):
         scal_flat[i]=flat[i][0]

     chunks=np.array_split(scal_flat,num_workers)
     
     
else:
    data=None
    chunks=None 
    

chunks = comm.scatter(chunks,root=0)
start = MPI.Wtime()
frequency(chunks)
end = MPI.Wtime()
final = comm.reduce(c, MPI.SUM, root=0)
#print (final)
if worker==0:
    
    print("Runtime", end-start);
#comm.barrier()

#print(freq)



#newData = comm.gather(data,root=0)
#
#
#if worker == 0:
#   print ('master:',newData)



