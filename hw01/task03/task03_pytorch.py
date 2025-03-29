
import torch
import torch.nn as nn

torch.set_printoptions(precision=4)

def cross_entropy_loss(output,target):
    return -(target*torch.log(output)).sum()

class TwoLayerNet(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(TwoLayerNet,self).__init__()
        self.fc1=nn.Linear(input_size,hidden_size,bias=False)
        self.fc1.activation=nn.ReLU()
        self.fc2=nn.Linear(hidden_size,output_size,bias=False)
        self.fc2.activation=nn.Softmax(dim=1)
   
    def forward(self,x):
        x=self.fc1.activation(self.fc1(x))
        x=self.fc2.activation(self.fc2(x))
        return x


# Hyperparameters
input_size=3
hidden_size=4
output_size=2
learning_rate=0.01
num_epochs=100

# Generate data
x1=torch.tensor([1.0,2.0,3.0])
x2=torch.tensor([4.0,5.0,6.0])
x=torch.stack([x1,x2])

# Generate weights
w1=torch.tensor([[0.1,0.2,0.3,0.4],[0.5,0.6,0.7,0.8],[0.9,1.0,1.1,1.2]],dtype=torch.float32)
w2=torch.tensor([[0.2,0.1],[0.4,0.5],[0.6,0.2],[0.8,0.7]],dtype=torch.float32)

# Generate target
y1=torch.tensor([0.0,1.0],dtype=torch.float32)
y2=torch.tensor([1.0,0.0],dtype=torch.float32)
target=torch.stack([y1,y2])

# Instantiate the model, set weight of model
model=TwoLayerNet(input_size,hidden_size,output_size)
with torch.no_grad():
    model.fc1.weight.copy_(w1.t())
    model.fc2.weight.copy_(w2.t())
criterion=cross_entropy_loss
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)

# Training loop
for epoch in range(num_epochs):
    # Forward pass
    outputs=model(x)
    loss=criterion(outputs,target)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Print the gradient of w1 and w2     
print("Gradient of w1: ",model.fc1.weight.grad.t())
print("Gradient of w2: ",model.fc2.weight.grad.t())

print("Output: ",outputs)