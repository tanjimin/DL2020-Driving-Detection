import os
import cv2
import torch
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

import sys
sys.path.append('test')
from helper import compute_ts_road_map
from validation import validation_loop
from utils import *

def epoch_loop(param):
    for epoch in range(param['epochs']):
        train_loop(epoch, param) 
        print("Epoch {}, Train Loss: {}".format(epoch, param['running_loss'] / len(param['train_loader'])))
        if epoch % 10 == 0:
            #validation_loop(epoch, param)
            save_path = 'saves_{}'.format(param['run_name'])
            if not os.path.exists(save_path):
                os.mkdir(save_path)

            else:
                if param['run_name'] in ['mosaic', 'camerabased_full_obj']:
                    torch.save(param['model'][0], 
                            '{}/static_{}'.format(save_path, epoch))
                    torch.save(param['model'][1], 
                            '{}/fusion_{}'.format(save_path, epoch))
                elif param['run_name'] in ['polar', 'front', 'camerabased', 'bbox','bbox_reg', 'bbox_reg_full']:
                    torch.save(param['model'], 
                            '{}/static_{}_{}'.format(save_path, param['run_name'], epoch))
                elif param['run_name'] in ['camerabased_full']:
                    torch.save(param['model'][1], 
                            '{}/FusionSixCameras_{}'.format(save_path, epoch))
                elif param['run_name'] in ['camerabased_unet']:
                    torch.save(param['model'][1], 
                            '{}/FusionUnet_{}'.format(save_path, epoch))
                else:
                    raise Error("Param['run name'] not found. Model cannot be saved.")

def train_loop(epoch, param):
    param['running_loss'] = 0.0 
    for batch_i, batch in enumerate(tqdm(param['train_loader'])):
        train(epoch, batch_i, batch, param)

def train(epoch, batch_i, batch, param):
    # if param['run_name'] != 'camerabased_full':
    #     param['optimizer'].zero_grad()
    # elif param['run_name'] == 'camerabased_full':
    #     param['optimizer'][0].zero_grad()
    #     param['optimizer'][1].zero_grad()
    param['optimizer'][0].zero_grad()
    param['optimizer'][1].zero_grad()
    outputs = None
    
    if not param['run_name'] in ['bbox_reg', 'bbox_reg_full']:
        inputs, labels = batch 
        inputs = inputs.to(param['device'])
        labels = labels.to(param['device'])
    else:
        inputs, samples, labels, graph = batch
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
        classifier = param['model'][2].train()
        camera_feature = camera_model(inputs).unsqueeze(1)
        bbox_feature = bbox_model(samples.view(-1, 8)) # Batched bbox
        camera_feature_batch = camera_feature.repeat(1, samples.shape[1], 1) # repeat to match num of bbox features
        outputs = classifier(camera_feature_batch.view(-1, 1024), bbox_feature).view(-1, samples.shape[1])
        if batch_i % 500 == 0:
            gen_bbox_heatmap_train(param, graph, camera_feature, epoch, batch_i)
    elif param['run_name'] == 'bbox_reg_full':
        camera_model = param['model'][0].train()
        bbox_model = param['model'][1].train()
        classifier = param['model'][2].train()
        # inputs of shape batch * 6 * 256 * 16 * 20
        camera_inputs = inputs.view(-1, 256, 16, 20)
        camera_feature = camera_model(camera_inputs).unsqueeze(1)
        bbox_feature = bbox_model(samples.view(-1, 8))
        camera_feature_batch = camera_feature.repeat(1, samples.shape[2], 1)
        outputs = classifier(camera_feature_batch.view(-1, 1024), bbox_feature).view(-1, samples.shape[1]).view(4, 6, 400)

    elif param['run_name'] in ['camerabased_full','camerabased_full_obj']:
        inputs = inputs.view(-1, 256, 16, 20)
        labels = labels.view(-1, 800, 800)

        static_camerabased = param['model'][0].train()
        fusion_cameras = param['model'][1].train()
        
        # [batch*6, 400, 538]
        outputs_cameras = static_camerabased(inputs).squeeze(1)
        # [batch, 6, 400, 538]
        outputs_cameras = outputs_cameras.view(-1, 6, 400, 538)
        # [batch, 800, 800]
        outputs = fusion_cameras(outputs_cameras).squeeze(1)

    elif param['run_name'] == 'camerabased_unet':
        inputs = inputs.view(-1, 256, 16, 20)
        labels = labels.view(-1, 800, 800)

        static_camerabased = param['model'][0]
        unet_cameras = param['model'][1].train()

        # [batch*6, 400, 538]
        outputs_cameras = static_camerabased(inputs).squeeze(1)
        # [batch, 6, 400, 538]
        outputs_cameras = outputs_cameras.view(-1, 6, 400, 538)
        # [batch, 1, 800, 800]
        unet_input = concat_cameras(outputs_cameras.cpu().numpy()).unsqueeze(1).to(param['device'])
        # [batch, 800, 800]
        outputs = unet_cameras(unet_input).squeeze(1)

    loss = param['criterion'](outputs, labels.float())

    loss.backward()
    param['optimizer'][0].step()
    param['optimizer'][1].step()
    param['running_loss'] += loss.item()
    
    if epoch % 5 == 1 and batch_i % 100 == 1 and param['run_name'] != 'bbox_reg':
        sample_path = 'sample_output_{}'.format(param['run_name'])
        if not os.path.exists(sample_path): 
            os.mkdir(sample_path)
        else:
            rand_camera = random.randrange(0,outputs.shape[0])
            cv2.imwrite('{}/sample_{}_{}_{}.png'.format(sample_path, 
                                                    epoch, 
                                                    batch_i,
                                                    rand_camera), 
                       outputs[rand_camera].detach().cpu().numpy() * 255)

def gen_bbox_heatmap_train(param, ground_truth, camera_feature, epoch, batch):
    camera_feature = camera_feature[:, 0, :]
    batch_size = camera_feature.shape[0]
    all_bbox = param['train_loader'].dataset.box_sampler.get_bbox().copy()
    all_bbox[:,:,1] = all_bbox[:,:,1] - 269 
    all_bbox_coord = all_bbox.mean(axis = 1)
    all_bbox_size = all_bbox.shape[0]
    bbox_model = param['model'][1].eval()
    classifier = param['model'][2].eval()
    model_input = torch.LongTensor(all_bbox).view(-1, 8).to(param['device']) / 10
    bbox_feature = bbox_model(model_input.float())
    for camera_batch in range(camera_feature.shape[0]):
        camera_batches = camera_feature[camera_batch].unsqueeze(0).repeat(all_bbox_size, 1)
        pred = classifier(camera_batches, bbox_feature).squeeze(1)
        max_value = pred.max().item()
        min_value = pred.min().item()
        fig, ax = plt.subplots()
        ax.scatter(all_bbox_coord[:, 0], all_bbox_coord[:, 1], c = pred.detach().cpu().numpy(), s = 1, edgecolor='')
        ax.set_aspect('equal')
        img_dir = 'bbox_imgs'
        if not os.path.exists(img_dir): os.mkdir(img_dir)
        plt.savefig('{}/train_img_{}_{}_max{}_min{}.png'.format(img_dir, epoch, batch, max_value, min_value))
        plt.close()
        cv2.imwrite('{}/train_img_{}_{}_label.png'.format(img_dir, epoch, batch), ground_truth[camera_batch].numpy() * 255)
        break

