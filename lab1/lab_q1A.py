#Author
#Istiaque Mannafee shaikat-303527

#Collaboration with:
#Paweena Tarepakdee -303405


from mpi4py import MPI
import numpy as np


N = 10**7
comm = MPI.COMM_WORLD
worker = comm.Get_rank()
num_workers = comm.Get_size()
N_worker = round(N/num_workers)
chunks = (N_worker*worker,N_worker*(worker+1))




def create_vector():
    V1 = np.random.rand(N)
    V2 = np.random.rand(N)
    data = (V1,V2)
    return data

def add_vector(V1,V2):
    start = MPI.Wtime()
    vC = []
    vA = V1[chunks[0]:chunks[1]] #making chunks of vectors
    vB = V2[chunks[0]:chunks[1]]
    for i in range(len(vA)):
        vC.append(vA[i]+vB[i])
    time_diff = MPI.Wtime() - start #google
    return vC,time_diff

def sort_vector(z): #took help from stack overflow
    V3 = []
    tokens = sorted(z, key=lambda x: x['worker'])
    for item in tokens: V3 += item['vC']
    print(V3) #final result
    print('execution time', exe_time) 


if worker != 0:
    data = comm.recv()
    vC,time_diff = add_vector(*data)
    data = {'vC':vC,'t_taken':time_diff, 'worker': worker}
    comm.send(data, dest=0) 
else:
    data=create_vector()
    for i in range(1,num_workers):
        comm.send(data, dest=i)
    vC,time_diff =add_vector(*data)
    dic = {'vC':vC,'t_taken':time_diff, 'worker': worker}
    temp = []
    exe_time = 0
    for w in range(1,num_workers):
        data = comm.recv() 
        temp.append(data) 
        exe_time += data['t_taken'] 
    temp.append(dic) 
    exe_time += time_diff 
    sort_vector(temp)    

   
