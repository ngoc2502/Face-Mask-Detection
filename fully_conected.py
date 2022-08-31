import numpy as np

from activations import *

def compute_nnet_outputs(Ws, X, need_all_layer_outputs):
    '''
    Computes the outputs of Neural Net by forward propagating X through the net.
    
    Parameters
    ----------
    Ws : list of numpy arrays
        Ws[l-1] is W of layer l with l >= 1 (layer 0 is input layer, 
        it doesn't have W); W of layer l will have the shape of 
        (d^(l-1)+1, d^(l)), where  d^(l-1) is the number of neurons 
        (not count the +1 neuron) of layer l-1 and  d^(l) is the number of 
        neurons (not count the +1 neuron) of layer l.
    X : numpy array, shape (N, d+1)
        The matrix of input vectors (each row corresponds to an input vector); 
        the first column of this matrix is all ones (corresponding to x_0).
    need_all_layer_outputs : bool
        If this var is true, we'll return a list of layer's-outputs (we'll 
        need this list when training); otherwise, we'll return the final 
        layer's output.
    Returns
    -------
    If `need_all_layer_outputs` is false, return
        A : numpy array, shape (N, K=10)
            The maxtrix of output vectors of final layer; each row is an 
            output vector (containing each class's probability given the 
            corresponding input vector).
    Else, return
        As : list of numpy arrays
            As[l] is the matrix of output vectors of layer l (l=0 will 
            correspond to input layer); each row is an output vector 
            (corresponding to an input vector); if layer l is not the final 
            layer, the first column of this matrix is all ones.
    '''    
    As=[X]
    A=X
    for i in Ws:
        A=Sigmoi.forward(A.dot(i))
        A=np.append(np.array([[1]]*len(A)),A,axis=1)
        As.append(A)
    As[len(As)-1]=np.delete(As[len(As)-1],0,1)

    if (need_all_layer_outputs):
        return As
    else:
        return As[len(As)-1]


def train_nnet(X, Y, val_X, val_Y, 
               hid_layer_sizes, 
               mb_size, learning_rate, max_epoch):
    '''
    Trains Neural Net on the dataset (X, Y); also prints out mean binary error 
    (the percentage of misclassified data points) on training set and 
    validation set during training
    Cost function: mean cross-entropy error
    Optimization algorithm: SGD
    
    Parameters
    ----------
    X : numpy array, shape (N, d + 1)
        The matrix of input vectors (each row corresponds to an input vector); 
        the first column of this matrix is all ones (corresponding to x_0).
    Y : numpy array, shape (N,) 
        The vector of outputs.
    val_X, val_Y : validation data, similar to X and Y
    hid_layer_sizes : list
        The list of hidden layer sizes; e.g., hid_layer_sizes = [20, 10] means
        the Net has 2 hidden layers, the 1st one has 20 neurons, and the 2nd 
        one has 10 neurons (not count the +1 neurons).
    mb_size : int
        Minibatch size of SGD.
    learning_rate : float
        Learning rate of SGD.
    max_epoch : int
        After this number of epochs, we'll terminate SGD.

    Returns
    -------
    (Ws, costs, errs) : tuple
        Ws : list of numpy arrays
            Ws[l-1] is W of layer l with l >= 1 (layer 0 is input layer, 
            it doesn't have W); W of layer l will have the shape of 
            (d^(l-1)+1, d^(l)), where d^(l-1) is the number of neurons 
            (not count the +1 neuron) of layer l-1 and d^(l) is the number of 
            neurons (not count the +1 neuron) of layer l.
        costs : list, len = max_epoch
            The list of costs after each epoch.
        errs : list, len = max_epoch
            The list of mean binary errors (on the training set) after each epoch.
    '''
    # Prepare for training
    K = len(np.unique(Y)) # Num classes

    layer_sizes = [X.shape[1] - 1] + hid_layer_sizes + [K]

    Ws = [np.random.randn(layer_sizes[i] + 1, layer_sizes[i + 1]) 
          / np.sqrt(layer_sizes[i] + 1) 
          for i in range(len(layer_sizes) - 1)] # formula to init Ws

    one_hot_Y = np.zeros((len(Y), K))
    one_hot_Y[np.arange(len(Y)), Y] = 1

    errs = [] # To save mean binary errors on training set during training
    val_errs = [] # To save mean binary errors on validation set during training
    N = len(X) # Num training pairs
    rnd_idxs = np.arange(N) # Random indexes    

    # Train
    for epoch in range(max_epoch):
        np.random.shuffle(rnd_idxs)
        for start_idx in range(0, N, mb_size):
            # Get minibach
            mb_X = X[rnd_idxs[start_idx:start_idx+mb_size]]
            mb_Y = one_hot_Y[rnd_idxs[start_idx:start_idx+mb_size]]
            
            # Forward-prop
            As = compute_nnet_outputs(Ws, mb_X, True)

            # Back-prop; on the way, compute each layer's gradient and update its W
            delta= As[len(As)-1] - mb_Y
            
            if start_idx == N- N % mb_size:
                grad=As[len(As)-2].T.dot(delta) /(N % mb_size)
            else:
                grad=As[len(As)-2].T.dot(delta) /mb_size
            
            # if N % mb_size != 0 
            # => last minibatch's size < mb_size
            # => when computing grad for last minibatch, 
            #  need to divide by last minibatch's size instead of mb_size
            
            Ws[-1] -= learning_rate * grad
            for i in range(2, len(Ws) + 1):
                #Compute delta's of layer l from delta's of layer l+1 (on a mini-batch)
                delta = delta.dot(Ws[-i + 1].T[:, 1:]) * As[-i][:, 1:] * (1 - As[-i][:, 1:])
                #Compute gradient of layer l from delta of layer l and outputs of layer l-1 (on a mini-batch)
                batch=mb_size
                if start_idx == N - N % mb_size:
                    if (N % mb_size !=0 ):
                        batch = N % mb_size
                        
                grad=As[len(As)-i-1].T.dot(delta) /batch
                
                Ws[-i] -= learning_rate * grad

        # Compute training info, save it, and print it
        A = compute_nnet_outputs(Ws, X, False)
        err = np.mean(np.argmax(A, axis=1) != Y) * 100
        errs.append(err)
        val_A = compute_nnet_outputs(Ws, val_X, False)
        val_err = np.mean(np.argmax(val_A, axis=1) != val_Y) * 100
        val_errs.append(val_err)
        print('Epoch %d, train err %.3f%%, val err %.3f%%' %(epoch, err, val_err))
            
    return Ws, errs, val_errs

