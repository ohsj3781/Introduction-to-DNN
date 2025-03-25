import numpy as np


def cross_entropy_loss(output,target):
    return -(target*np.log(output)).sum(dim=1).mean()

x1=np.array([1.0,2.0,3.0])
x2=np.array([4.0,5.0,6.0])

x=np.stack([x1,x2])

w1=np.array([[0.1,0.2,0.3,0.4],[0.5,0.6,0.7,0.8],[0.9,1.0,1.1,1.2]],dtype=np.float32)
w2=np.array([[0.2,0.1],[0.4,0.5],[0.6,0.2],[0.8,0.7]],dtype=np.float32)

hidden_layer=np.maximum(np.dot(x,w1),0)

output_Layer=np.exp(np.dot(hidden_layer,w2))/np.sum(np.exp(np.dot(hidden_layer,w2)),axis=1,keepdims=True)