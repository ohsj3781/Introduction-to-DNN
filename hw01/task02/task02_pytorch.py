import torch
import torch.nn as nn
import torch.nn.functional as F

# Generate data
x1=torch.tensor([0.1,0.2,0.3],dtype=torch.float32)
x2=torch.tensor([0.4,0.5,0.6],dtype=torch.float32)
x=torch.stack([x1,x2])

w1=torch.tensor([[0.1,0.2,0.3,0.4],[0.5,0.6,0.7,0.8],[0.9,1.0,1.1,1.2]],dtype=torch.float32,requires_grad=True)
w2=torch.tensor([[0.2,0.1],[0.4,0.5],[0.6,0.2],[0.8,0.7]],dtype=torch.float32,requires_grad=True)

hidden_layer=F.relu(torch.matmul(x,w1))

output_layer=F.softmax(torch.matmul(hidden_layer,w2),dim=1)

target=torch.tensor([[0,1],[1,0]],dtype=torch.float32)

loss=-(target*torch.log(output_layer)).sum(dim=1).mean()


loss.backward()

print("Gradient of w1: ",w1.grad)

