import random
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from PIL import Image
from torchvision import transforms
from torchvision.datasets import DatasetFolder
from torchvision.models.resnet import resnet18

from utilspretext import (AverageMeter, Logger, Memory, ModelCheckpoint,
                   NoiseContrastiveEstimator, Progbar, pil_loader)



negative_nb = 100 # number of negative examples in NCE
checkpoint_dir = 'jigsaw_models'
log_filename = 'pretraining_log_jigsaw'


def epoch_loop(params):
    memory = Memory(size = len(params['train_loader'].dataset), weight= 0.5, device = params['device'])
    memory.initialize(params['model'], params['train_loader'])

    checkpoint = ModelCheckpoint(mode = 'min', directory = checkpoint_dir )
    logger = Logger(log_filename)
    
    loss_weight = 0.5

    for epoch in range(params['epochs']):
        print('\nEpoch: {}'.format(epoch))
        memory.update_weighted_count()
        train_loss = AverageMeter('train_loss')
        #bar = Progbar(len(params['train_loader'].dataset), stateful_metrics=['train_loss', 'valid_loss'])
        
        for step, batch in enumerate(tqdm(params['train_loader'])): 
            # prepare batch
            images = batch['original'].to(params['device'])
            patches = [element.to(params['device']) for element in batch['patches']]
            index = batch['index']
            representations = memory.return_representations(index).to(params['device']).detach()
            # zero grad
            params['optimizer'].zero_grad()
            
            #forward, loss, backward, step
            output = params['model'](images = images, patches = patches, mode = 1)
            
            loss_1 = params['criterion'](representations, output[1], index, memory, negative_nb = negative_nb)
            loss_2 = params['criterion'](representations, output[0], index, memory, negative_nb = negative_nb) 
            loss = loss_weight * loss_1 + (1 - loss_weight) * loss_2
            
            loss.backward()
            params['optimizer'].step()
            
            #update representation memory
            memory.update(index, output[0].detach().cpu().numpy())
            
            # update metric and bar
            train_loss.update(loss.item(), images.shape[0])
            #bar.update(step, values=[('train_loss', train_loss.return_avg())])
        logger.update(epoch, train_loss.return_avg())

        #save model if improved
        checkpoint.save_model(params['model'], train_loss.return_avg(), epoch)
