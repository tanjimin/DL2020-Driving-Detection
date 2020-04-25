import torch
from tqdm import tqdm

def epoch_loop(param):
    for epoch in range(param['epochs']):
       batch_loop(epoch, param) 

def batch_loop(epoch, param):
    param['running_loss'] = 0.0 
    for batch_i, batch in enumerate(tqdm(param['train_loader'])):
        train(batch_i, batch, param)

def train(batch_i, batch, param):
    inputs, labels = batch 
    param['optimizer'].zero_grad()
    outputs = param['model'](inputs)
    loss = param['criterion'](outputs, labels)
    loss.backward()
    param['optimizer'].step()
    param['running_loss'] += loss.item()
    if batch_i % 1000 == 1:
        print("Loss is: ", param['running_loss'] / batch_i)
