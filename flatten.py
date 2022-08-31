import numpy as np

def Flatten(A:np.array):
    A=A.flatten()
    return  A

def Flatten_backprop(A,prev_size):
    A=A.reshape(prev_size)
    return A

# A= np.array([[1,2,3,4],[0,0,4,5],[2,2,2,5],[0,9,8,1]])
# print(Flatten(A))
# size=4,4
# print(Flatten_backprop(A,size))