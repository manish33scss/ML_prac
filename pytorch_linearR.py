# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 14:53:01 2019

@author: Manish
"""
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt

x_val=[i for i in range(11)]
print("val of x ",x_val)

x_train=np.asarray(x_val, dtype=np.float32)
x_train.shape
x_train=x_train.reshape(-1,1)
x_train.shape
#y=2x+1 -- need to madel this relationship

y_val=[2*i +1 for i in x_val]
y_val=np.asarray(y_val, dtype=np.float32)
y_val.shape
y_val=y_val.reshape(-1,1)
y_val.shape

class LRmodel(nn.Module):
    def __init__(self,input_size,output_size):
        super(LRmodel,self).__init__()
        self.linear=nn.Linear(input_dim,output_dim)
        
    def forward(self,x):
        out=self.linear(x)
        return out
input_dim=1
output_dim=1
model=LRmodel(input_dim,output_dim)
criterion=nn.MSELoss()
l_r=0.01
optimiser=torch.optim.SGD(model.parameters(),lr=l_r)
epoch=100
for epoch in range(epoch):
    epoch+=1
    inputs=Variable(torch.from_numpy(x_train))
    labels=Variable(torch.from_numpy(y_val))
    optimiser.zero_grad()
    outputs=model(inputs)
    loss=criterion(outputs,labels)
    loss.backward()
    optimiser.step()
    print("epoch{}, loss{}".format(epoch,loss.item()))

predicted=model(Variable(torch.from_numpy(x_train))).data.numpy()
plt.plot(x_train, y_val,'go',label='True data', alpha=0.5)
plt.plot(x_train,predicted,'--',label='Predicted',alpha=0.5)
plt.legend(loc='best')
plt.show()
