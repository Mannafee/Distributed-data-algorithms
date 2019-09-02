from mpi4py import MPI
import pandas as pd
import numpy as np

from scipy.spatial import distance





k=5     



comm = MPI.COMM_WORLD
num_workers = comm.Get_size()
worker= comm.Get_rank()


def initial_centroid(data):
    centroids={i:[np.random.randint(0,len(data))]
    for i in range(k)
            }
    for i in range(k):
        centroids[i]=data[centroids[i][0]]
    
#    print(centroids[1])
    return centroids


def k_mean(chunk,centroids):
    dst=[[0 for x in range (k)]for y in range(len(chunk))]
    cluster=[[0 for x in range (len(chunk))]for y in range(k)]
    centers_new=[0 for x in range(k)]

    for x in range (len(chunk)):
        for y in range (k):
            dst[x][y]=distance.euclidean(centroids[y],chunk[x])
#        print( dst[x])
    for x in range (len(chunk)):
        for y in range (k):            
            cluster[np.argmin(dst[x])][x]=chunk[x] 
            
#    print(cluster[0])       
    for i in range(k):
        
        centers_new[i] = np.mean(cluster[i])
   
    vC[worker]=centers_new
#    print(vC)
    return vC
    
    
def global_centroid(new):
     for i in range(k):
         for x in range(num_workers):
             if(new[x]!=0):
                vD[i] =np.mean(new[x][i])
        
#     print(vD)
    
    
def collect_data():    
    df=pd.read_csv("Absenteeism_at_work.csv",sep=';')
    df=df.values
#    print(data)
    return df



if worker== 0:
   data=collect_data()
   ini_cent=initial_centroid(data)
   chunks=np.array_split(data,num_workers)
   centers_global=[0 for x in range(k)]
   new_c=[0 for i in range (num_workers)]
else:
    data=None
    chunks=None
    ini_cent=None
    centers_global=None
    new_c=None
    

vA=comm.scatter(chunks,root=0)
vB=comm.bcast(ini_cent,root=0)
vC=comm.bcast(new_c,root=0)
vD=comm.bcast(centers_global,root=0)
#data=comm.bcast(data,root=0)
#print(worker,vA)
start = MPI.Wtime()
new=k_mean(vA,vB)
end = MPI.Wtime()
#print(new[1])

if worker==0:
    a=global_centroid(new)
    print("Runtime", end-start);
