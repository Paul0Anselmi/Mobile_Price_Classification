import numpy as np 

def relu(z):
    return np.maximum(0,z)

def relu_derivete(z):
    return (z>0).astype(float)

def sigmoid(z):
    z =np.clip(z,-500,500)
    return 1.0/(1.0+np.exp(z))

def softmax(z):
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

