import numpy as np

np.set_printoptions(precision=4)

def relu(x):
    return np.maximum(x, 0)

def relu_grad(x):
    return np.where(x > 0, 1, 0)

def softmax(x):
    e_x=np.exp(x)
    return e_x/np.sum(e_x, axis=1, keepdims=True)

def cross_entropy_loss(output,target):
    return -(target*np.log(output)).sum()
class TwoLayerNet:
    def __init__(self,input_size,hidden_size,output_size):
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.w=[np.random.rand(input_size,hidden_size),np.random.rand(hidden_size,output_size)]

    def forward(self,x):
        z1=np.dot(x,self.w[0])
        a1=relu(z1)
        z2=np.dot(a1,self.w[1])
        outputs=softmax(z2)
        cache={
            'X':x,
            'z1':z1,
            'a1':a1,
            'z2':z2,
            'outputs':outputs
        }
        return outputs,cache
    def backward(self,cache,target):
        X   = cache['X']
        z1  = cache['z1']
        a1  = cache['a1']
        z2  = cache['z2']
        outputs = cache['outputs']
        m   = X.shape[0]  # batch_size

        dz2=(outputs-target)

        dw2=np.dot(a1.T,dz2)

        da1=np.dot(dz2,self.w[1].T)

        dz1=da1*relu_grad(z1)

        dw1=np.dot(X.T,dz1)
       
        grads = {
            'W1': dw1,
            'W2': dw2,
        }
        return grads    

# Hyperparameters
input_size=3
hidden_size=4
output_size=2

# Generate data
x1=np.array([1.0,2.0,3.0])
x2=np.array([4.0,5.0,6.0])
x=np.stack([x1,x2])

# Generate weights
w1=np.array([[0.1,0.2,0.3,0.4],[0.5,0.6,0.7,0.8],[0.9,1.0,1.1,1.2]],dtype=np.float32)
w2=np.array([[0.2,0.1],[0.4,0.5],[0.6,0.2],[0.8,0.7]],dtype=np.float32)
w=np.array([w1,w2],dtype=object)

# Generate target
y1=np.array([0.0,1.0],dtype=np.float32)
y2=np.array([1.0,0.0],dtype=np.float32)
targets=np.stack([y1,y2])

# Instantiate the model, set weight of model
model=TwoLayerNet(input_size,hidden_size,output_size)
model.w[0]=w1
model.w[1]=w2

# Forward pass
outputs,cache=model.forward(x)
loss=cross_entropy_loss(outputs,targets)

grad=model.backward(cache,targets)
print("Gradient of w1: ",grad['W1'])
