from tqdm import tqdm
from torchvision import datasets, transforms
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

train_losses = []
#test_losses = []
train_acc = []
#test_acc = []
lrs = []

def train11(model, device, train_loader, optimizer, epoch, criterion, scheduler):
  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  trn_corr = 0
  
  
  import torch
  import torch.nn as nn
  import torch.nn.functional as F
  import torch
  
  # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
  criterion = nn.CrossEntropyLoss()
  
  criterion.cuda()
  
  #!pip install torchsummary
  #from torchsummary import summary
  #use_cuda = torch.cuda.is_available()
  #device = torch.device("cuda" if use_cuda else "cpu")
  #print(device)
   
  for batch_idx, (data, target) in enumerate(pbar):
    # get samples
    data, target = data.to("cuda"), target.to("cuda")

    # Init
    optimizer.zero_grad()
    # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
    # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

    # Predict
    y_pred = model(data)
    loss = criterion(y_pred, target)
  

    # Calculate loss
    #loss = F.nll_loss(y_pred, target)
    train_losses.append(loss)

    # Backpropagation
    loss.backward()
    optimizer.step()
    
    if scheduler:
      scheduler.step()
      lrs.append(scheduler.get_last_lr()[0])

    # Update pbar-tqdm
    
    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)

    pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    train_acc.append(100*correct/processed)

def train12(model, device, train_loader, optimizer, epoch, criterion):
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    trn_corr = 0
    
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch
    
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    
    criterion.cuda()
    
    #!pip install torchsummary
    #from torchsummary import summary
    #use_cuda = torch.cuda.is_available()
    #device = torch.device("cuda" if use_cuda else "cpu")
    #print(device)
    
    for batch_idx, (data, target) in enumerate(pbar):
      # get samples
      data, target = data.to("cuda"), target.to("cuda")

      # Init
      optimizer.zero_grad()
      # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
      # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

      # Predict
      y_pred = model(data)
      loss = criterion(y_pred, target)
    

      # Calculate loss
      #loss = F.nll_loss(y_pred, target)
      train_losses.append(loss)

      # Backpropagation
      loss.backward()
      optimizer.step()
      

      # Update pbar-tqdm
      
      pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
      correct += pred.eq(target.view_as(pred)).sum().item()
      processed += len(data)

      pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
      train_acc.append(100*correct/processed)