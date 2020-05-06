import os
import cv2
import torch
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('test')
from helper import compute_ts_road_map

def validation_loop(epoch, param):
    param['running_loss'] = 0.0 
    for batch_i, batch in enumerate(tqdm(param['validation_loader'])):
        validation(epoch, batch_i, batch, param)
    print("Epoch: {}, Val loss: {}".format(epoch, param['running_loss'] / 
                                       len(param['validation_loader'])))

def validation(epoch, batch_i, batch, param):
    with torch.no_grad():
        outputs = None
        if param['run_name'] != 'bbox_reg':
            inputs, labels = batch 
            inputs = inputs.to(param['device'])
            labels = labels.to(param['device'])
        else:
            inputs, samples, labels, graph = batch
            inputs = inputs.to(param['device'])
            labels = labels.to(param['device'])
            samples = samples.to(param['device'])

        if param['run_name'] == 'mosaic':
            fusion_layer = param['model'][0].eval()
            static_model = param['model'][1].eval()
            fusion_outputs = fusion_layer(inputs)
            outputs = static_model(fusion_outputs).squeeze(1)
        elif param['run_name'] == 'polar':
            static_polar = param['model'].eval()
            outputs = static_polar(inputs.squeeze(1)).squeeze(1)
        elif param['run_name'] in ['front','bbox']:
            static_front = param['model'].eval()
            outputs = static_front(inputs).squeeze(1)
            for i in range(outputs.shape[0]):
                ts = compute_ts_road_map(outputs[i], labels[i])
                param['ts'] += ts
        elif param['run_name'] in ['camerabased']:
            inputs = inputs.view(-1, 256, 16, 20)
            labels = labels.view(-1, 400, 538)
            static_camerabased = param['model'].eval()
            outputs = static_camerabased(inputs).squeeze(1)
        elif param['run_name'] == 'bbox_reg':
            camera_model = param['model'][0].eval()
            bbox_model = param['model'][1].eval()
            classifier = param['model'][2].eval()
            camera_feature = camera_model(inputs).unsqueeze(1)
            bbox_feature = bbox_model(samples.view(-1, 8)) # Batched bbox
            camera_feature_batch = camera_feature.repeat(1, samples.shape[1], 1).view(-1, 1024) # repeat to match num of bbox features
            bbox_feature_positive = bbox_feature[labels.reshape(-1) == 1]
            camera_feature_batch_positive = camera_feature_batch[labels.reshape(-1) == 1]
            labels_positive = labels.reshape(-1)[labels.reshape(-1) == 1]
            outputs = classifier(camera_feature_batch_positive, bbox_feature_positive).squeeze(1)
            gen_bbox_heatmap(param, graph, camera_feature, epoch, batch_i)

        loss = param['criterion'](outputs, labels_positive.float())
        param['running_loss'] += loss.item()

def gen_bbox_heatmap(param, ground_truth, camera_feature, epoch, batch):
    camera_feature = camera_feature[:, 0, :]
    batch_size = camera_feature.shape[0]
    all_bbox = param['validation_loader'].dataset.box_sampler.get_bbox() - 269
    all_bbox_coord = all_bbox.mean(axis = 1)
    all_bbox_size = all_bbox.shape[0]
    bbox_model = param['model'][1].eval()
    classifier = param['model'][2].eval()
    model_input = torch.LongTensor(all_bbox).view(-1, 8).to(param['device'])
    bbox_feature = bbox_model(model_input.float())
    for camera_batch in range(camera_feature.shape[0]):
        camera_batches = camera_feature[camera_batch].unsqueeze(0).repeat(all_bbox_size, 1)
        pred = classifier(camera_batches, bbox_feature).squeeze(1)
        max_value = pred.max().item()
        min_value = pred.min().item()
        fig, ax = plt.subplots()
        ax.scatter(all_bbox_coord[:, 0], all_bbox_coord[:, 1], c = pred.detach().cpu().numpy(), s = 1, edgecolor='')
        img_dir = 'bbox_imgs'
        if not os.path.exists(img_dir): os.mkdir(img_dir)
        plt.savefig('{}/test_img_{}_{}_max{}_min{}.png'.format(img_dir, epoch, batch, max_value, min_value))
        plt.close()
        cv2.imwrite('{}/test_img_{}_{}_label.png'.format(img_dir, epoch, batch), np.rot90(ground_truth[camera_batch].numpy() * 255))
        break

