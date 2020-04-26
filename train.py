import os
import cv2
import torch
from tqdm import tqdm

def epoch_loop(param):
    for epoch in range(param['epochs']):
        batch_loop(epoch, param) 
        if epoch % 10 == 0:
            if not os.path.exists('saves'):
                os.mkdir('saves')
            else:
                torch.save(param['model'][0], 'saves/fusion_' + str(epoch))
                torch.save(param['model'][1], 'saves/static_' + str(epoch))


def batch_loop(epoch, param):
    param['running_loss'] = 0.0 
    for batch_i, batch in enumerate(tqdm(param['train_loader'])):
        train(epoch, batch_i, batch, param)

def train(epoch, batch_i, batch, param):
    fusion_layer = param['model'][0]
    static_model = param['model'][1]
    inputs, labels = batch 
    inputs = inputs.to(param['device'])
    labels = labels.to(param['device'])
    param['optimizer'].zero_grad()
    fusion_outputs = fusion_layer(inputs)
    outputs = static_model(fusion_outputs).squeeze(1)
    #import pdb; pdb.set_trace()
    loss = param['criterion'](outputs, labels)
    loss.backward()
    param['optimizer'].step()
    param['running_loss'] += loss.item()
    if epoch % 5 == 1 and batch_i % 100 == 1:
        print("Loss is: ", param['running_loss'] / batch_i)
        if not os.path.exists('sample_output'): 
            os.mkdir('sample_output')
        else:
            cv2.imwrite('sample_output/sample_{}_{}.png'.format(epoch, 
                                                                batch_i), 
                        outputs[0].detach().cpu().numpy() * 255)
