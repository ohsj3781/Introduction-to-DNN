# README.md 2020314916_승재_오
##  To implement cross_entropy_loss function using pytorch and numpy we can use the following codes.

### pytorch version
````
def cross_entropy_loss_torch(output,target):
    return -torch.sum(target*torch.log(output))
````

### numpy version
````
def cross_entropy_loss_numpy(output,target):
    return -np.sum(target*np.log(output))
````