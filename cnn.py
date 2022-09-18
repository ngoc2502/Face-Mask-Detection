from array import array
from turtle import forward
import numpy as np


def conv_step(k,b,a_slice_pre):
            """
            Arguments:
            a_slice -- slice of input data of shape (f, f,n_kernel)
            W -- Weight parameters contained in a window - matrix of shape (f, f,n_kernel)
            
            Returns:
            Z -- a scalar value
            """
            # Element-wise
            s = np.multiply(a_slice_pre, k)
            Z = np.sum(s)
            #for Bias =1
            Z = Z + b
            return Z

class convolution():
        def __init__(self,A,kernel,padding,strike,b):
            # n_ker is the number of kernel applided for A 
            # numpy arr of shape (f,f,n_preKer,n_ker)
            self.kernel=kernel
            # padding for input 
            self.padding=padding
            self.strike=strike
            # output of previous activation layer shape (batch,Height,Width,n_ker) 
            self.A_pre=A
            #bias
            self.b=b

        def zero_padding(self):
                '''
                Arguments:
                A: numpy Array python shape(H,W) where Heigh h, Width W

                Returns:
                A_pad: (H+2*pad,W+2*pad)
                '''
                A_pad = np.pad(self.A_pre,((0,0),(self.padding,self.padding),(self.padding,self.padding),(0,0)),'constant',constant_values=0)
                return A_pad

        def conv_step(w,a_slice_pre):
            """
            Arguments:
            a_slice -- slice of input data of shape (f, f,n_kernel)
            W -- Weight parameters contained in a window - matrix of shape (f, f,n_kernel)
            
            Returns:
            Z -- a scalar value
            """
            # Element-wise
            s = np.multiply(a_slice_pre,w)
            Z = np.sum(s)
            #for Bias =1
            Z = Z + 1.0
            return Z

        def forward(self):
            """
            Forward propagation for a convolution  
            Arguments:
            Returns:
            Z -- conv output, numpy array of shape (batch,Height,Width,n_Ker)
            
            batch for the numbers of input images
            """
            (batch,Height,Width,n_kernel_pre) = self.A_pre.shape
            (f, f,_,n_ker) = self.kernel.shape
            # Compute the dimensions of the CONV output volume 
            n_H = int((Height - f + 2 * self.padding) / self.strike) + 1
            n_W = int((Width - f + 2 * self.padding) / self.strike) + 1

            # Initialize the output volume Z with zeros.
            Z = np.zeros((batch,n_H, n_W,n_ker))
            
            # Create A_prev_pad by padding A_prev
            A_prepaded = self.zero_padding()
            
            # loop over the batch of training examples
            for i in range(batch):                              
                a_prev_pad = A_prepaded[i]                             
                for h in range(n_H):                          
                    for w in range(n_W):   
                        for j in range(n_ker):  
                            # Find the corners 
                            y_start = h * self.strike
                            y_end = y_start + f
                            x_start = w * self.strike
                            x_end = x_start + f
                            
                            a_slice_prev = a_prev_pad[y_start:y_end, x_start:x_end,:]
                            Z[i,h,w,j] = conv_step(self.kernel[...,j],self.b[...,j],a_slice_prev) 
            
            cache=self.A_pre,self.kernel 
            return Z,cache # catch A for backprop
    
        def backward(self,dZ, cache):
            """
            Arguments:
            dZ -- gradient of the cost with respect to the output of the conv layer (Z), numpy array of shape (batch, n_H, n_W, n_C)
            cache -- cache of values from conv_forward()

            Returns:
            dA_prev -- gradient of the cost with respect to the input of the conv layer (A_prev),
                    numpy array of shape (batch, n_H_prev, n_W_prev, n_C_prev)
            dW -- gradient of the cost with respect to the weights of the conv layer (W)
                numpy array of shape (f, f, n_C_prev, n_C)
            db -- gradient of the cost with respect to the biases of the conv layer (b)
                numpy array of shape (1, 1, 1, n_C)
            """
            
            (A_prev, W) = cache
            (batch, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape            
            (f, f, n_C_prev, n_C) = W.shape            
            (batch, n_H, n_W, n_C) = dZ.shape
            
            # Initialize dA_prev, dW, db with the correct shapes
            dA_prev = np.zeros((batch, n_H_prev, n_W_prev, n_C_prev))                           
            dW = np.zeros((f, f, n_C_prev, n_C))
            db = np.zeros((1, 1, 1, n_C))

            A_prev_pad = self.zero_padding()
            dA_prev_pad = self.zero_padding()
            
            for i in range(batch):                       
                # loop over the training examples   
                # select ith training example from A_prev_pad and dA_prev_pad
                a_prev_pad = A_prev_pad[i]
                da_prev_pad = dA_prev_pad[i]
                
                for h in range(n_H):                   # loop over vertical axis of the output volume
                    for w in range(n_W):               # loop over horizontal axis of the output volume
                        for c in range(n_C ):           # loop over the channels of the output volume  
                            # Find the corners of the current "slice"
                            vert_start = h
                            vert_end = vert_start + f
                            horiz_start = w
                            horiz_end = horiz_start + f
                            
                            # Use the corners to define the slice from a_prev_pad
                            a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                            # Update gradients for the window and the filter's parameters 
                            # Ws=W[:,:,:,c].shape
                            # Wchamc=W[...,c].shape
                            # dZs=dZ[i, h, w, c]
                            # das=da_prev_pad[vert_start:vert_end, horiz_start:horiz_end,:].shape
                            da_prev_pad[vert_start:vert_end, horiz_start:horiz_end,:] += W[...,c] * dZ[i, h, w, c]
                           
                            dW[:,:,:,c] += a_slice * dZ[i, h, w, c]
                            db[:,:,:,c] += dZ[i, h, w, c] 

                # Set the ith training example's dA_prev to the unpaded da_prev_pad 
                # dAs=dA_prev[i, :, :, :].shape
                # dA_pS=dA_prev[i, :, :, :].shape
                # da_pre_s=da_prev_pad[self.padding:-self.padding, self.padding:-self.padding, :].shape
                #dA_prev[i,:,:,:]=da_prev_pad[self.padding:-self.padding, self.padding:-self.padding, :]
                dA_prev[i, :, :, :] = da_prev_pad[:,:,:]
                # Making sure output shape is correct
            assert(dA_prev.shape == (batch, n_H_prev, n_W_prev, n_C_prev))
            return dA_prev, dW, db    

#   TEST FORWARD AND BACKWARD FUNCT 
np.random.seed(1)
A = np.random.randn(4,3,3,2)
kernel = np.random.randn(3,3,2,5)
b=np.ones((1,1,1,5))
size=A.shape
C=convolution(A,kernel,0,1,b)
Z,cache=C.forward()
Da,Dk,db=C.backward(Z,cache)

print(Da)
print(Da.shape)
print(Dk)
print(Dk.shape)

# TEST CONV_STEP _ PASS
# kernel=np.array([[1,0,1],[0,1,0],[1,0,1]])
# A_slide=np.array([[1,0,0],[1,1,0],[1,1,1]])
# def conv_step(k,a_slice_pre):
#             """
#             Arguments:
#             a_slice -- slice of input data of shape (f, f,n_kernel)
#             W -- Weight parameters contained in a window - matrix of shape (f, f,n_kernel)
            
#             Returns:
#             Z -- a scalar value
#             """
#             # Element-wise
#             s = np.multiply(a_slice_pre, k)
#             Z = np.sum(s)
#             #for Bias =1
#             Z = Z + 1.0
#             return Z
# print (A_slide)
# print("===========================")
# print(kernel)
# z=conv_step(A_slide,kernel)
# print(z)