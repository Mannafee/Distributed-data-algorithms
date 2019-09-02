from mpi4py import MPI
import numpy as np

N = 10**5
comm = MPI.COMM_WORLD
worker = comm.Get_rank()
num_workers = comm.Get_size()

def createarray():
    vA=np.random.randint(10,size=N)
    return vA

#comm.barrier()


def NsendAll(vB):
   
    for i in range(1,num_workers):
        comm.send(vB, dest=i)
        
    


if worker==0:
    vA=createarray()
    start = MPI.Wtime()
    NsendAll(vA)
    end=MPI.Wtime()
    print("Runtime", end-start);
   
    
else:
    data=comm.recv()
    
comm.barrier()
    