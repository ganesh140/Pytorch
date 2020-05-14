#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch, torchvision
T = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
mnist_data = torchvision.datasets.MNIST('mnist_data',transform=T,download=True)

mnist_dataloader = torch.utils.data.DataLoader(mnist_data,batch_size=128)


# In[6]:


from torch import nn
class Mnet(nn.Module):
    def __init__(self):
        super(Mnet,self).__init__()
        self.linear1 = nn.Linear(28*28,100)
        self.linear2 = nn.Linear(100,50)
        self.final_linear = nn.Linear(50,10)
        
        self.relu = nn.ReLU()
        
    def forward(self,images):
        x = images.view(-1,28*28)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.final_linear(x)
        return x


# In[1]:


from torch import optim

import visdom 
from torch.autograd import Variable
model = Mnet()
cec_loss = nn.CrossEntropyLoss()
params = model.parameters()
optimizer = optim.Adam(params=params,lr=0.001)

n_epochs=3
n_iterations=0

vis=visdom.Visdom()
vis_window=vis.line(np.array([0]),np.array([0]))

for e in range(n_epochs):
    for i,(images,labels) in enumerate(mnist_dataloader):
        images = Variable(images)
        labels = Variable(labels)
        output = model(images)

        model.zero_grad()
        loss = cec_loss(output,labels)
        loss.backward()

        optimizer.step()
        n_iterations+=1

        vis.line(np.array([loss.item()]),np.array([n_iterations]),win=vis_window,update='append')
        


# In[ ]:




