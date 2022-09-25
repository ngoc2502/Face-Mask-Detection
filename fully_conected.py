import numpy as np

class fully_connected:
    def __init__(self, input, output_size):
        batch,input_size=input.shape
        np.random.seed(1)
        self.weights = np.random.rand(batch,input_size, output_size).round(2)
        self.bias = np.random.rand(batch,1, output_size).round(2)
        self.input = input
        self.output = None 
        self.batch=batch
        self.input_size=input_size
        
    def forward(self):
        self.output=np.zeros((self.batch,self.weights.shape[2]))
        for i in range(self.batch):
            tmp=self.input[i,:].reshape((1,len(self.input[i,:])))
            self.output[i,:]=np.dot(tmp,self.weights[i,:])
        return self.output

    def backward(self, output_err, learning_rate):

        input_error=np.zeros(self.input.shape)
        d_Weights=np.zeros(self.weights.shape)
        d_bias=np.zeros(self.bias.shape)

        for i in range(self.batch):
            input_error[i,:]=np.dot(output_err[i,:],self.weights[i,:].T)

            tmp_input=self.input[i,:].reshape((len(self.input[i,:]),1))
            tmp_output=output_err[i].reshape(1,(len(output_err[i])))
            
            d_Weights[i,:,:]=np.dot(tmp_input,tmp_output)
            d_bias[i,:,:]= tmp_output
            
            self.weights[i,:] -= learning_rate*d_Weights[i,:]
            self.bias[i,:,:] -= learning_rate*d_bias[i,:,:]
        return input_error 
        
    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        d_weights = np.dot(self.input.T, output_error)
        d_bias = output_error
        self.weights -= learning_rate * d_weights
        self.bias -= learning_rate * d_bias
        return input_error 


np.random.seed(1)
A = np.random.randn(5,30)
# output_error=np.random.randn(5,2)
# learning_rate=0.1

F=fully_connected(A,2)
f=F.forward()
print(f)
# b=F.backward(output_error,learning_rate)

# print(b)
# print(b.shape)

# np.random.seed(1)
# A = np.array([[1,2,0,1,3]]) 
# error = np.array([[0.09,0.08]]) 
# F = fully_connected(5,2) 
# outFor = F.forward_propagation(A) 
# outBack = F.backward_propagation(error,0.1) 
# print(outBack)
