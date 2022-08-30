import numpy as np

class Activation():
    def __init__(self,activ_for,activ_back):
        self.activation=activ_for
        self.activ_backprop=activ_back

    def forward(self,input):
        self.input=input
        return self.activation()

    def backward(self):
        #
        pass


class Relu(Activation):
    def __init__(self):
        def ReLu_forward(x):
            '''
            forward ReLu activation give the output=max(0,x)
            '''
            return np.maximum(0,x) 

        def ReLu_backprop(x):
            res=0
            if x<=0:
                return res
            else:
                return 1
        super().__init__(ReLu_forward,ReLu_backprop)


class Sigmoi(Activation):
    def __init__(self):
        def sigmoi_forward(x):
            return 1/1+np.exp(-x)
                
        def sigmoi_backward(x):
            return sigmoi_forward(x)*(1-sigmoi_forward(x))
        
        super().__init__(sigmoi_forward,sigmoi_backward)


    
    

