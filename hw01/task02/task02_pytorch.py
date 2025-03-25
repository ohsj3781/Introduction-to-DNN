import torch
import torch.nn as nn
import torch.nn.functional as F

gpu_avail=torch.cuda.is_available()
print("GPU available: ",gpu_avail)
if gpu_avail:
    device=torch.device("cuda")
else:
    device=torch.device("cpu")
print("Device: ",device)

def cross_entropy_loss(output,target):
    return -(target*torch.log(output)).sum(dim=1).mean()

# Generate data
x1=torch.tensor([0.1,0.2,0.3],dtype=torch.float32)
x2=torch.tensor([0.4,0.5,0.6],dtype=torch.float32)
x=torch.stack([x1,x2])

w1=torch.tensor([[0.1,0.2,0.3,0.4],[0.5,0.6,0.7,0.8],[0.9,1.0,1.1,1.2]],dtype=torch.float32,requires_grad=True)
w2=torch.tensor([[0.2,0.1],[0.4,0.5],[0.6,0.2],[0.8,0.7]],dtype=torch.float32,requires_grad=True)

hidden_layer=F.relu(torch.matmul(x,w1))


print(torch.matmul(hidden_layer,w2))

output_layer=F.softmax(torch.matmul(hidden_layer,w2),dim=1)

# print("Output layer: ",output_layer)

target=torch.tensor([[0,1],[1,0]],dtype=torch.float32)
print(target[0])
loss=cross_entropy_loss(output_layer,target)


loss.backward()

print("Gradient of w1: ",w1.grad)

