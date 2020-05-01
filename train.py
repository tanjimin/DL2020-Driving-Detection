import os
import cv2
import torch
from tqdm import tqdm

import sys
sys.path.append('test')
from helper import compute_ts_road_map
from validation import validation_loop

def epoch_loop(param):
    for epoch in range(param['epochs']):
        train_loop(epoch, param) 
        if epoch % 10 == 0:
            validation_loop(epoch, param)
            save_path = 'saves_{}'.format(param['run_name'])
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            else:
                if param['run_name'] == 'mosaic':
                    torch.save(param['model'][0], 
                            '{}/fusion_{}'.format(save_path, epoch))
                    torch.save(param['model'][1], 
                            '{}/static_{}'.format(save_path, epoch))
                elif param['run_name'] in ['polar', 'front', 'camerabased', 'bbox','bbox_reg']:
                    torch.save(param['model'], 
                            '{}/static_{}_{}'.format(save_path, param['run_name'], epoch))
                else:
                    raise Error("Param['run name'] not found. Model cannot be saved.")

def train_loop(epoch, param):
    param['running_loss'] = 0.0 
    for batch_i, batch in enumerate(tqdm(param['train_loader'])):
        train(epoch, batch_i, batch, param)

def train(epoch, batch_i, batch, param):
    param['optimizer'].zero_grad()
    outputs = None
    if param['run_name'] != 'bbox_reg':
        inputs, labels = batch 
        inputs = inputs.to(param['device'])
        labels = labels.to(param['device'])
    else:
        inputs, samples, labels, _ = batch
        inputs = inputs.to(param['device'])
        labels = labels.to(param['device'])
        samples = samples.to(param['device'])

    if param['run_name'] == 'mosaic':
        fusion_layer = param['model'][0].train()
        static_model = param['model'][1].train()
        fusion_outputs = fusion_layer(inputs)
        outputs = static_model(fusion_outputs).squeeze(1)
    elif param['run_name'] == 'polar':
        static_polar = param['model'].train()
        outputs = static_polar(inputs.squeeze(1)).squeeze(1)
    elif param['run_name'] in ['front', 'bbox']:
        static_front = param['model'].train()
        outputs = static_front(inputs).squeeze(1)
    elif param['run_name'] in ['camerabased']:
        inputs = inputs.view(-1, 256, 16, 20)
        labels = labels.view(-1, 400, 538)
        static_camerabased = param['model'].train()
        outputs = static_camerabased(inputs).squeeze(1)
    elif param['run_name'] == 'bbox_reg':
        camera_model = param['model'][0].train()
        bbox_model = param['model'][1].train()
        camera_feature = camera_model(inputs)
        bbox_feature = bbox_model(samples.view(-1, 8)) # Batched bbox
        camera_feature_batch = camera_feature.repeat(1, samples.shape[1], 1).view(-1, 32) # repeat to match num of bbox features
        #print(camera_feature_batch.shape, )
        
    if param['run_name'] != "bbox_reg":
        loss = param['criterion'](outputs, labels.float())
    else:
        loss = param['criterion'](camera_feature_batch, bbox_feature, labels.view(-1,1))
    loss.backward()
    param['optimizer'].step()
    param['running_loss'] += loss.item()
    if epoch % 5 == 1 and batch_i % 100 == 1:
        print("Epoch {}, Loss: {}".format(epoch, param['running_loss'] / batch_i))
        sample_path = 'sample_output_{}'.format(param['run_name'])
        #if not os.path.exists(sample_path): 
        #    os.mkdir(sample_path)
        #else:
        #    cv2.imwrite('{}/sample_{}_{}.png'.format(sample_path, 
        #                                             epoch, 
        #                                             batch_i), 
        #                outputs[0].detach().cpu().numpy() * 255)

