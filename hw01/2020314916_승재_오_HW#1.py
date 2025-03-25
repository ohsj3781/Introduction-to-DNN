import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def cross_entropy_loss_torch(output,target):
    return -torch.sum(target*torch.log(output))

gpu_avail=torch.cuda.is_available()
print("GPU available: ",gpu_avail)
if gpu_avail:
    device=torch.device("cuda")
else:
    device=torch.device("cpu")

print("Torch version")
# Torch version
# Generate data
x1=torch.tensor([0.1,0.2,0.3],dtype=torch.float32)
x2=torch.tensor([0.4,0.5,0.6],dtype=torch.float32)
x=torch.stack([x1,x2])

# Generate weights
w1=torch.tensor([[0.1,0.2,0.3,0.4],[0.5,0.6,0.7,0.8],[0.9,1.0,1.1,1.2]],dtype=torch.float32,requires_grad=True)
w2=torch.tensor([[0.2,0.1],[0.4,0.5],[0.6,0.2],[0.8,0.7]],dtype=torch.float32,requires_grad=True)

hidden_layer=F.relu(torch.matmul(x,w1))

output_layer=F.softmax(torch.matmul(hidden_layer,w2),dim=1)

print("Output layer: ",output_layer.data)

y1=torch.tensor([0,1],dtype=torch.float32)
y2=torch.tensor([1,0],dtype=torch.float32)
target=torch.stack([y1,y2])

loss=cross_entropy_loss_torch(output_layer,target)
# print("Loss: ",loss.data)

loss.backward()
print("Gradient of w1: ",w1.grad.data)
print("Gradient of w2: ",w2.grad.data)

num_epochs=100
learning_rate=0.1

for epoch in range(num_epochs):
    hidden_layer=F.relu(torch.matmul(x,w1))
    output_layer=F.softmax(torch.matmul(hidden_layer,w2),dim=1)
    loss=cross_entropy_loss_torch(output_layer,target)
    # print("Epoch:%d, Gradient of w1: %s"%(epoch,w1.grad.data))
    loss.backward()
    with torch.no_grad():
        w1-=learning_rate*w1.grad
        w2-=learning_rate*w2.grad
        w1.grad.zero_()
        w2.grad.zero_()

print("Output layer after training: ",output_layer.data)
print("Loss after training: ",loss.data)


# print("Numpy version")
# # Numpy version
# def cross_entropy_loss_numpy(output,target):
#     return -np.sum(target*np.log(output))

# # Generate data
# x1=np.array([0.1,0.2,0.3],dtype=np.float32)
# x2=np.array([0.4,0.5,0.6],dtype=np.float32)
# x=np.stack([x1,x2])

# # Generate weights
# w1=np.array([[0.1,0.2,0.3,0.4],[0.5,0.6,0.7,0.8],[0.9,1.0,1.1,1.2]],dtype=np.float32)
# w2=np.array([[0.2,0.1],[0.4,0.5],[0.6,0.2],[0.8,0.7]],dtype=np.float32)

# hidden_layer=np.maximum(np.matmul(x,w1),0)

# output_layer=np.exp(np.matmul(hidden_layer,w2))/np.sum(np.exp(np.matmul(hidden_layer,w2)),axis=1,keepdims=True)

# print("Output layer: ",output_layer)

# y1=np.array([0,1],dtype=np.float32)
# y2=np.array([1,0],dtype=np.float32)
# target=np.stack([y1,y2])

# loss=cross_entropy_loss_numpy(output_layer,target)
# print("Loss: ",loss)

# grad_output=output_layer-target

# def compute_gradients_numpy(x, w1, w2, target):
#     # Forward pass
#     hidden_layer = np.maximum(np.matmul(x, w1), 0)
#     output_layer = np.exp(np.matmul(hidden_layer, w2)) / np.sum(np.exp(np.matmul(hidden_layer, w2)), axis=1, keepdims=True)
    
#     # Compute loss gradient w.r.t. output
#     grad_output = output_layer - target
    
#     # Backpropagate to w2
#     grad_w2 = np.matmul(hidden_layer.T, grad_output)
    
#     # Backpropagate to hidden layer
#     grad_hidden = np.matmul(grad_output, w2.T)
#     grad_hidden[hidden_layer <= 0] = 0  # ReLU derivative
    
#     # Backpropagate to w1
#     grad_w1 = np.matmul(x.T, grad_hidden)
    
#     return grad_w1, grad_w2

# grad_w1, grad_w2 = compute_gradients_numpy(x, w1, w2, target)
# print("Gradient of w1: ", grad_w1)
# print("Gradient of w2: ", grad_w2)




