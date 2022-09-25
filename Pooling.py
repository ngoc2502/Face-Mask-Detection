import numpy as np

class Pooling():
    def __init__(self, strike,size):
        self.strike=strike
        self.size=size        

    def max_pool2d_forward(self,A):
        '''
        Forward propagation for a pooling, help reduce computation
            Arguments:
                A -- output of the previous layers, SHAPE(batch,height,width,n_chanel)
            Returns:
                Z -- pooling output, numpy array\n 
                batch -- for the numbers of input images
        '''

        (batch,Height,Width,n_chanel)=A.shape
        (f,f)=self.size
        # compute the dimensions of Pooling output volumn
        n_H= int((Height-f)/self.strike)+1
        n_W=int((Width-f)/self.strike)+1

        # Initialize the output volumn Z with zeros
        Z = np.zeros((batch,n_H,n_W,n_chanel))

        # loop over the batch of training
        for i in range(batch):
            for h in range(n_H):
                for w in range(n_W):
                    for c in range(n_chanel):
                        # Find current position of kernel
                        y_s=h*self.strike
                        y_e=y_s+f
                        x_s=w*self.strike
                        x_e=x_s+f
                        a_slice=A[i,y_s:y_e,x_s:x_e,:]
                        Z[i,h,w,c]=np.max(a_slice)   
        cache=A
        return Z, cache

    def mean_pool2d_forward(self,A,batch):
        '''
        Forward propagation for a pooling, help reduce computation\n
        Arguments:
            A -- output of the previous layers, SHAPE(batch,height,width,n_chanel)
        Returns:
            Z -- pooling output, numpy array of shape \n
            batch -- the numbers of input images

        '''
        (Height,Width)=A.shape
        (f,f)=self.size
        # compute the dimensions of Pooling output volumn
        n_H= int((Height-f)/self.strike)+1
        n_W=int((Width-f)/self.strike)+1
        # Initialize the output volumn Z with zeros
        Z = np.zeros((n_H,n_W))

        # loop over the batch of training
        for i in range(batch):
            for h in range(n_H):
                for w in range(n_W):
                    # Find current position of kernel
                    y_s=h*self.strike
                    y_e=y_s+f
                    x_s=w*self.strike
                    x_e=x_s+f
                    a_slice=A[x_s:x_e,y_s:y_e]
                    Z[h,w]=np.mean(a_slice)
        return Z

    def pool_backward(self,dA, cache):
        """
        Implements the backward
            Arguments:
                dA -- gradient of cost with respect to the output of the pooling layer, same shape as A\n
                cache -- cache output from the forward pass of the pooling layer, contains the layer's input and hparameters 
            Returns:
                dA_prev -- gradient of cost with respect to the input of the pooling layer, same shape as A_prev
        """
        A_prev= cache        
        
        (m, n_H, n_W, n_C) = dA.shape
        (f,f)=self.size
        dA_prev = np.zeros(A_prev.shape)
        
        for i in range(m):                      
            a_prev = A_prev[i]
            for h in range(n_H):                  
                for w in range(n_W):              
                    for c in range(n_C):          
                        vert_start = h
                        vert_end = vert_start + f
                        horiz_start = w
                        horiz_end = horiz_start + f
                        # Define the current slice from a_prev 
                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                        # Create the mask from a_prev_slice
                        mask = (a_prev_slice==np.max(a_prev_slice))
                        # Set dA_prev to be dA_prev + (the mask multiplied by the correct entry of dA) 
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += np.multiply(mask, dA[i, h, w, c])
        return dA_prev

    def mean_pool2d_backward():
        pass

# np.random.seed(1)
# A_pre = np.random.randn(5,5,3,2)
# print(A_pre.shape)
# print(A_pre)

# print('**************************************************')
# P=Pooling(1,(2,2))
# resForW,cache=P.max_pool2d_forward(A_pre)
# print(resForW)
# print(resForW.shape)
# dA=np.random.randn(5,4,2,2)
# resBackW=P.pool_backward(dA,cache)

# print('**************************************************')
# print(resBackW)
# print(resBackW.shape)

'''
convert:
1 3 9 1 
1 3 4 2
1 3 3 2 
2 2 2 0
'''
