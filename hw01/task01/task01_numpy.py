import numpy as np

np.set_printoptions(precision=4)

def relu(x):
    return np.maximum(x, 0)

def softmax(x):
    e_x=np.exp(x)
    return e_x/np.sum(e_x, axis=1, keepdims=True)
class TwoLayerNet:
    def __init__(self,input_size,hidden_size,output_size):
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.w=[np.random.rand(input_size,hidden_size),np.random.rand(hidden_size,output_size)]

    def forward(self,x):
        hidden_layer=relu(np.dot(x,self.w[0]))
        outputs=softmax(np.dot(hidden_layer,self.w[1]))
        return outputs

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

# Instantiate the model, set weight of model
model=TwoLayerNet(input_size,hidden_size,output_size)
model.w[0]=w1
model.w[1]=w2

# Forward pass
outputs=model.forward(x)

print("Outputs ",outputs)