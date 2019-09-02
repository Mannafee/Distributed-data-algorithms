from mpi4py import MPI
import numpy as np

N = 10**5
comm = MPI.COMM_WORLD
#P = 2**comm.Get_rank()
worker=comm.Get_rank()
num_workers = comm.Get_size()



def createarray():
    vA=np.random.randint(10,size=N)
    return vA

#comm.barrier()



def EsendAll():   
    recvProc=int((worker-1)/2)
    destA=2*worker+1 
    destB=2*worker+2
#    print(destA)
#    print(destB)
    data=comm.recv(source=recvProc)
     
    if destA<num_workers:
        comm.send(data,dest=destA)
    if destB<num_workers:
        comm.send(data,dest=destB)


comm.barrier()

start = MPI.Wtime()      
if worker==0:
    vQ=createarray()
    comm.send(vQ, dest=1)
    comm.send(vQ, dest=2)
    end=MPI.Wtime()
    print('time',end - start)
else:
    EsendAll()
    

#comm.barrier()
#time=MPI.Wtime() - start
#print(time)




    
#comm.finalize()