# Input layer with 3 nodes
# Hidden layer with 4 nodes and ReLU activation
# Output layer with 2 nodes and softmax activation

import torch
import torch.nn as nn
import torch.nn.functional as F

# Generate intputData
x1=torch.tensor([1.0,2.0,3.0])
x2=torch.tensor([4.0,5.0,6.0])

x=torch.stack([x1,x2])

# Generate weights
w1=torch.tensor([[0.1,0.2,0.3,0.4],[0.5,0.6,0.7,0.8],[0.9,1.0,1.1,1.2]],dtype=torch.float32)
w2=torch.tensor([[0.2,0.1],[0.4,0.5],[0.6,0.2],[0.8,0.7]],dtype=torch.float32)


# Generate hidden layer
hidden_layer=F.relu(torch.matmul(x,w1))

output_layer=F.softmax(torch.matmul(hidden_layer,w2),dim=1)


print("Output layer: ",output_layer)