import numpy as np

def Flatten_step(A):
    A=A.flatten()
    return  A

def Flatten_backprop_step(A,prev_size):
    A=A.reshape(prev_size)
    return A     

def Flatt_forward(A):
    (batch, height, width, n_chanel)=A.shape
    a_flat=np.zeros((batch,height*width*n_chanel))
    for i in range(batch):
        a_pre=A[i]
        a_flat[i,:]=Flatten_step(a_pre)
    
    cache_size= height, width, n_chanel
    return a_flat,cache_size
       
def Flatt_backward(A,pre_size):
    (batch,size)=A.shape
    height,width,n_chanel=pre_size
    back_prop_A=np.zeros((batch,height,width,n_chanel))
    for i in range (batch):
        back_prop_A[i,:,:,:]=Flatten_backprop_step(A[i,:],pre_size)
    return back_prop_A

np.random.seed(1)
A_pre = np.random.randn(5,5,3,2)
a_flat,cache_size= Flatt_forward(A_pre)
a_back=Flatt_backward(a_flat,cache_size)
print(A_pre==a_back)
