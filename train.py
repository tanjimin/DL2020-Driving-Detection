import os
import cv2
import torch
from tqdm import tqdm

def epoch_loop(param):
    for epoch in range(param['epochs']):
        batch_loop(epoch, param) 
        if epoch % 10 == 0:
            save_path = 'saves_{}'.format(param['run_name'])
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            else:
                torch.save(param['model'][0], 
                           '{}/fusion_{}'.format(save_path, epoch))
                torch.save(param['model'][1], 
                           '{}/static_{}'.format(save_path, epoch))

def batch_loop(epoch, param):
    param['running_loss'] = 0.0 
    for batch_i, batch in enumerate(tqdm(param['train_loader'])):
        train(epoch, batch_i, batch, param)

def train(epoch, batch_i, batch, param):
    inputs, labels = batch 
    inputs = inputs.to(param['device'])
    labels = labels.to(param['device'])
    param['optimizer'].zero_grad()
    outputs = None
    if param['run_name'] == 'mosaic':
        fusion_layer = param['model'][0]
        static_model = param['model'][1]
        fusion_outputs = fusion_layer(inputs)
        outputs = static_model(fusion_outputs).squeeze(1)
    elif param['run_name'] == 'polar':
        static_polar = param['model']
        outputs = static_polar(inputs.squeeze(1)).squeeze(1)
    #import pdb; pdb.set_trace()
    loss = param['criterion'](outputs, labels)
    loss.backward()
    param['optimizer'].step()
    param['running_loss'] += loss.item()
    if epoch % 5 == 1 and batch_i % 100 == 1:
        print("Epoch {}, Loss: {}".format(epoch, param['running_loss'] / batch_i))
        sample_path = 'sample_output_{}'.format(param['run_name'])
        if not os.path.exists(sample_path): 
            os.mkdir(sample_path)
        else:
            cv2.imwrite('{}/sample_{}_{}.png'.format(sample_path, 
                                                     epoch, 
                                                     batch_i), 
                        outputs[0].detach().cpu().numpy() * 255)
