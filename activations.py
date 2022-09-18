import numpy as np

class Relu():
        def __init__(self,x):
            self.x=x
        def ReLu_forward(self):
            '''
            forward ReLu activation give the output=max(0,x)
            '''
            return np.maximum(0,self.x) 

        def ReLu_backprop(self):
            res=0
            if self.x<=0:
                return res
            else:
                return 1


class Sigmoi():
        def __init__(self,x):
            self.x=x

        def sigmoi_forward(self):
            return 1/1+np.exp(-self.x)
                
        def sigmoi_backward(self):
            return self.sigmoi_forward()*(1-self.sigmoi_forward())
        
class softmax():
        def __init__(self,x:np.array):
            self.x=x

        def forward(self):
            sum=np.exp(self.x).sum()
            res=np.zeros(len(self.x))
            for i in range(len(self.x)):
                res[i]=np.exp(self.x[i])/sum
            return res
        
        def backward(self):
            I=np.eye(len(self.x))
            res=self.forward()*(I-self.forward())
            return res

a=[1,5.21,1.83,1.08,1.251,-2.56]
s=softmax(a)
f=s.forward()

b=s.backward()
print(f)
print('============================================================')
print(b)


    
    

