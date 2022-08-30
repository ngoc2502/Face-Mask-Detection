import numpy as np

class Pooling():
    def __init__(self, strike,size,type):
        self.strike=strike
        self.size=size        

    def max_pool2d_forward(self,A, batch):
        '''
        Forward propagation for a pooling, help reduce computation
        
        Arguments:
        A -- output of the previous layers
        Returns:
        Z -- pooling output, numpy array of shape 

        batch for the numbers of input images
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
                    Z[h,w]=np.max(a_slice)

        return Z

    
    def mean_pool2d_forward(self,A,batch):
        '''
        Forward propagation for a pooling, help reduce computation
        
        Arguments:
        A -- output of the previous layers
        Returns:
        Z -- pooling output, numpy array of shape 

        batch for the numbers of input images
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

    def max_pool2d_backward():
        '''
        Compute the cost for a certain filter and traing exemple
        '''
        
        pass

    def mean_pool2d_backward():
        pass




A=np.array([[1,1,1,2],[3,3,3,2],[9,4,3,2],[1,2,2,0]])
print (A)
P=Pooling(1,(2,2),max)
res=P.mean_pool2d_forward(A,2)
print(res)


'''
convert:

1 3 9 1 
1 3 4 2
1 3 3 2 
2 2 2 0
'''