from array import array
from turtle import forward
import numpy as np


class convolution():
        def __init__(self,kernel,padding:0,strike:0,outputSize):
            self.kernel=kernel
            self.padding=padding
            self.strike=strike
            self.outputSize=outputSize

        def zero_padding(self,A):
                '''
                A: numpy Array python shape(H,W) where Heigh h, Width W,
                return
                A_pad: (H+2*pad,W+2*pad)
                '''
                A_pad = np.pad(A,(self.padding))
                return A_pad
        def conv_step(self,a_slice, W):
            """
           
            Arguments:
            a_slice -- slice of input data of shape (f, f)
            W -- Weight parameters contained in a window - matrix of shape (f, f)
            b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)
            
            Returns:
            Z -- a scalar value
            """

            # Element-wise
            s = np.multiply(a_slice, W)
            Z = np.sum(s)
            Z = Z + 1.0
            return Z

        def forward(self,A, W):
            """
            Forward propagation for a convolution  
            Arguments:
            A -- output activations of the previous layer, numpy array of shape (m, H,w)
            W -- Weights, numpy array of shape (f, f)
            hparameters --dictionary ( stride,pad)
                
            Returns:
            Z -- conv output, numpy array of shape (m, H, W)
         
            """

            (m, H,W) = A.shape
            
            # Retrieve dimensions from W's shape
            (f, f) = W.shape
           
            # Compute the dimensions of the CONV output volume 
            n_H = int((H - f + 2 * self.padding) / self.stride) + 1
            n_W = int((W - f + 2 * self.padding) / self.stride) + 1

            # Initialize the output volume Z with zeros.
            Z = np.zeros((m, n_H, n_W))
            
            # Create A_prev_pad by padding A_prev
            A_pad = zero_padding(A, self.padding)
             # loop over the batch of training examples
            for i in range(m):                              
                a_prev_pad = A_pad[i]                             
                for h in range(n_H):                          
                    for w in range(n_W):                       
                            # Find the corners 
                            y_start = h * self.stride
                            y_end = y_start + f
                            x_start = w * self.stride
                            x_end = x_start + f
                            
                            a_slice_prev = a_prev_pad[y_start:y_end, x_start:x_end, :]
                            Z[i, h, w] = conv_step(a_slice_prev, W[:, :])
                            
            return Z       
          
                    



        
    

