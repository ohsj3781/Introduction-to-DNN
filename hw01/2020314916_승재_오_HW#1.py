import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def cross_entropy_loss_torch(output,target):
    return -torch.sum(target*torch.log(output))

print("Torch version")
print("Task 1")
# Torch version

# Generate data
x1=torch.tensor([1,2,3],dtype=torch.float32)
x2=torch.tensor([4,5,6],dtype=torch.float32)
x=torch.stack([x1,x2])

# Generate weights
w1=torch.tensor([[0.1,0.2,0.3,0.4],[0.5,0.6,0.7,0.8],[0.9,1.0,1.1,1.2]],dtype=torch.float32,requires_grad=True)
w2=torch.tensor([[0.2,0.1],[0.4,0.5],[0.6,0.2],[0.8,0.7]],dtype=torch.float32,requires_grad=True)

# Caculate hidden layer with relu activation
hidden_layer=F.relu(torch.matmul(x,w1))

# Calculate output layer with softmax activation
output_layer=torch.matmul(hidden_layer,w2)
output_layer=F.softmax(output_layer,dim=1)


# Print output layer
print("Output layer: ",output_layer.data)

print("Task 2")

y1=torch.tensor([0,1],dtype=torch.float32)
y2=torch.tensor([1,0],dtype=torch.float32)
y=torch.stack([y1,y2])

# Calculate loss
loss=cross_entropy_loss_torch(output_layer,y)
# Backpropagation
loss.backward()
print("loss :",loss)
print("Gradient of w1: ",w1.grad.data)

# print("Task 3")

# num_epochs=100
# learning_rate=0.1

# for epoch in range(num_epochs):
#     hidden_layer=F.relu(torch.matmul(x,w1))
#     output_layer=F.softmax(torch.matmul(hidden_layer,w2),dim=1)
#     loss=cross_entropy_loss_torch(output_layer,y)
#     # print("Epoch:%d, Gradient of w1: %s"%(epoch,w1.grad.data))
#     loss.backward()
#     with torch.no_grad():
#         w1-=learning_rate*w1.grad
#         w2-=learning_rate*w2.grad
#         w1.grad.zero_()
#         w2.grad.zero_()

# print("w1 after training: ",w1.data)
# print("w2 after training: ",w2.data)

# def cross_entropy_loss_numpy(output,target):
#     return -np.sum(target*np.log(output))

# #Numpy version
# print("Numpy version")
# print("Task 1")

# # Generate data
# x1=np.array([0.1,0.2,0.3],dtype=np.float32)
# x2=np.array([0.4,0.5,0.6],dtype=np.float32)
# x=np.stack([x1,x2])

# # Generate weights
# w1=np.array([[0.1,0.2,0.3,0.4],[0.5,0.6,0.7,0.8],[0.9,1.0,1.1,1.2]],dtype=np.float32)
# w2=np.array([[0.2,0.1],[0.4,0.5],[0.6,0.2],[0.8,0.7]],dtype=np.float32)

# # Caculate hidden layer with relu activation
# hidden_layer=np.maximum(np.matmul(x,w1),0)

# # Calculate output layer with softmax activation
# output_layer=np.exp(np.matmul(hidden_layer,w2))/np.sum(np.exp(np.matmul(hidden_layer,w2)),axis=1,keepdims=True)

# # Print output layer
# print("Output layer: ",output_layer)

# print("Task 2")

# # Generate target
# y1=np.array([0,1],dtype=np.float32)
# y2=np.array([1,0],dtype=np.float32)
# target=np.stack([y1,y2])

# # Calculate loss
# loss=cross_entropy_loss_numpy(output_layer,target)
# print(loss)


# grad_output=output_layer-target

# # def compute_gradients_numpy(x, w1, w2, target):
# #     # Forward pass
# #     hidden_layer = np.maximum(np.matmul(x, w1), 0)
# #     output_layer = np.exp(np.matmul(hidden_layer, w2)) / np.sum(np.exp(np.matmul(hidden_layer, w2)), axis=1, keepdims=True)
    
# #     # Compute loss gradient w.r.t. output
# #     grad_output = output_layer - target
    
# #     # Backpropagate to w2
# #     grad_w2 = np.matmul(hidden_layer.T, grad_output)
    
# #     # Backpropagate to hidden layer
# #     grad_hidden = np.matmul(grad_output, w2.T)
# #     grad_hidden[hidden_layer <= 0] = 0  # ReLU derivative
    
# #     # Backpropagate to w1
# #     grad_w1 = np.matmul(x.T, grad_hidden)
    
# #     return grad_w1, grad_w2

# grad_w1, grad_w2 = compute_gradients_numpy(x, w1, w2, target)
# print("Gradient of w1: ", grad_w1)
# print("Gradient of w2: ", grad_w2)




