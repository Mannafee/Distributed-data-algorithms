#Author
#Istiaque Mannafee shaikat-303527

#Collaboration with:
#Paweena Tarepakdee -303405


from mpi4py import MPI

import numpy as np

comm = MPI.COMM_WORLD
worker = comm.Get_rank() 
num_workers = comm.Get_size() 

N = 10**3
N_worker = int(N/num_workers) 
chunks = (N_worker*worker,N_worker*(worker+1))
#print(chunks) 

def sum_vector(V):
    start_t = MPI.Wtime()
    sum_v = sum(V[chunks[0]:chunks[1]])
    time_diff = MPI.Wtime() - start_t
    return sum_v,time_diff

def create_vector():
    V = np.random.rand(N)
#    print(V)
    data = V
    for i in range(1,num_workers):
        comm.send(data, dest=i)
    return data
    

if worker != 0: 
    data = comm.recv()
    sum_v,time_diff = sum_vector(data)
    data = {'sum':sum_v,'t_taken':time_diff}
    comm.send(data, dest=0) 
else:
    data=create_vector()
    sum_v,time_diff = sum_vector(data)
    
    t_sum = 0
    exe_time = 0
    for w in range(1,num_workers):
        data = comm.recv() 
        t_sum += data['sum']
        
        exe_time += data['t_taken'] 
       
    t_sum += sum_v 
    
    exe_time += time_diff 
    
#    print('average sum', t_sum/N) 
    print('execution time', exe_time) 


