import numpy as np
import torch
from torch import nn
from .common import CONFIG
class CompactCNN(nn.Module):
 """Small temporal CNN over 64 log-mel channels; no pretrained weights."""
 def __init__(self,mean=None,std=None):
  super().__init__()
  self.register_buffer('mean',torch.tensor(np.zeros((64,1),np.float32) if mean is None else mean,dtype=torch.float32))
  self.register_buffer('std',torch.tensor(np.ones((64,1),np.float32) if std is None else std,dtype=torch.float32))
  self.encoder=nn.Sequential(nn.Conv1d(64,32,5,padding=2),nn.BatchNorm1d(32),nn.ReLU(),nn.AvgPool1d(2),nn.Conv1d(32,32,3,padding=1),nn.BatchNorm1d(32),nn.ReLU(),nn.AvgPool1d(2),nn.Conv1d(32,48,3,padding=1),nn.BatchNorm1d(48),nn.ReLU())
  self.head=nn.Sequential(nn.Linear(96,32),nn.ReLU(),nn.Dropout(.2),nn.Linear(32,1))
 def forward(self,x):
  x=self.encoder((x-self.mean)/self.std)
  mu=x.mean(dim=2);var=((x-mu.unsqueeze(-1))**2).mean(dim=2)
  return self.head(torch.cat((mu,torch.sqrt(var+1e-5)),dim=1)).squeeze(1)
