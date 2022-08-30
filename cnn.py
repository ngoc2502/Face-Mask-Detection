from array import array
from turtle import forward
import numpy as np


class convolution():
    def __init__(self,kernel,padding:0,strike:0,outputSize):
        self.kernel=kernel
        self.padding=padding
        self.strike=strike
        self.outputSize=outputSize

        def zero_padding(A,pad):
                '''
                A: numpy Array python shape(H,W) where Heigh h, Width W,
                return
                A_pad: (H+2*pad,W+2*pad)
                '''
                A_pad = np.pad(A,(pad))
                return A_pad
        def conv_step(a_slice, W, b):
            """
            Apply one filter defined by parameters W on a single slice (a_slice) of the output activation 
            of the previous layer.
            
            Arguments:
            a_slice -- slice of input data of shape (f, f, n_C_prev)
            W -- Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
            b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)
            
            Returns:
            Z -- a scalar value, result of convolving the sliding window (W, b) on a slice x of the input data
            """

            # Element-wise
            s = np.multiply(a_slice, W)
            Z = np.sum(s)
            # Add bias b to Z. Cast b to a float() so that Z results in a scalar value.
            Z = Z + float(b)
            return Z

        def conv_forward(A, W, b, hparameters):
            """
            Forward propagation for a convolution
            
            Arguments:
            A -- output activations of the previous layer, numpy array of shape (m, H,w)
            W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
            b -- Biases, numpy array of shape (1, 1, 1, n_C)
            hparameters -- python dictionary containing "stride" and "pad"
                
            Returns:
            Z -- conv output, numpy array of shape (m, H, W)
            cache -- cache of values needed for the conv_backward() function
            """

            (m, H,W) = A.shape
            
            # Retrieve dimensions from W's shape (≈1 line)
            (f, f) = W.shape
            
            # Retrieve information from "hparameters" (≈2 lines)
            stride = hparameters['stride']
            pad = hparameters['pad']
            
            # Compute the dimensions of the CONV output volume using the formula given above. Hint: use int() to floor. (≈2 lines)
            n_H = int((H - f + 2 * pad) / stride) + 1
            n_W = int((W - f + 2 * pad) / stride) + 1
            
            # Initialize the output volume Z with zeros. (≈1 line)
            Z = np.zeros((m, n_H, n_W))
            
            # Create A_prev_pad by padding A_prev
            A_pad = zero_padding(A, pad)
            
            for i in range(m):                               # loop over the batch of training examples
                a_prev_pad = A_pad[i]                               # Select ith training example's padded activation
                for h in range(n_H):                          
                    for w in range(n_W):                       
                            
                            # Find the corners 
                            y_start = h * stride
                            y_end = y_start + f
                            x_start = w * stride
                            x_end = x_start + f
                            
                            a_slice_prev = a_prev_pad[y_start:y_end, x_start:x_end, :]
                            
                            # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron. (≈1 line)
                            #Z[i, h, w] = conv_step(a_slice_prev, W[...], b[...])
                                                
                    



        
    

