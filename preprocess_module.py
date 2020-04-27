import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import cv2
import numpy as np

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

def reorder_tensor(tensor):
    '''
    Input/Output tensor shape: [batch_size, 6(images per sample), 3, H, W]
    
    Original
    image_names = [
    'CAM_FRONT_LEFT.jpeg',
    'CAM_FRONT.jpeg',
    'CAM_FRONT_RIGHT.jpeg',
    'CAM_BACK_LEFT.jpeg',
    'CAM_BACK.jpeg',
    'CAM_BACK_RIGHT.jpeg',
    ]
    
    Output
    image_names = [
    'CAM_FRONT_LEFT.jpeg',
    'CAM_FRONT.jpeg',
    'CAM_FRONT_RIGHT.jpeg',
    'CAM_BACK_RIGHT.jpeg',
    'CAM_BACK.jpeg',
    'CAM_BACK_LEFT.jpeg'
    ]
    '''
    out_tensor = torch.zeros(tensor.shape)
    
    out_tensor[:,0,:] = tensor[:,0,:]
    out_tensor[:,1,:] = tensor[:,1,:]
    out_tensor[:,2,:] = tensor[:,2,:]
    out_tensor[:,3,:] = tensor[:,5,:]
    out_tensor[:,4,:] = tensor[:,4,:]
    out_tensor[:,5,:] = tensor[:,3,:]
    
    return out_tensor

def warpper(data, reshape_size):
    
    img = data.permute(1,2,0).numpy()
    img = (img * 255).astype(np.uint8)
    
    img = np.rot90(img)
    img = np.flip(img, 1)
    # (width, height * 1.5, 3)
    new_img = np.zeros((1836, 390, 3))
    # -height
    new_img[:, -256:, :] = img
    img = new_img
    img = cv2.resize(img, (reshape_size, reshape_size))

    polar_image = cv2.linearPolar(src=img, center=(1500, 1500), maxRadius=1500, flags=cv2.WARP_FILL_OUTLIERS + cv2.WARP_INVERSE_MAP)
    polar_image = polar_image.astype(np.uint8)
    
    return polar_image

def camera_preprocess(batch_sample):
    
    '''
    batch_sample/output shape: [batch_size, 6(images per sample), 3, H, W]
    '''

    batch_sample = reorder_tensor(batch_sample)
    
    return batch_sample

def polar_preprocess(batch_sample, reshape_size = 3000):

    '''
    preprocess for polar images
    output shape: [batch_size, 3, H, W]
    '''
    
    batch_sample = reorder_tensor(batch_sample)
    
    batch_n = batch_sample.shape[0]
    camera_n = batch_sample.shape[1]
    channel_n = batch_sample.shape[2]
    height = batch_sample.shape[3]
    width = batch_sample.shape[4]
    
    output_tensor = torch.ones((batch_n, 3, reshape_size, reshape_size))

    flatten_width = torch.zeros((batch_n, channel_n, height, width * camera_n))
    
    for i in range(camera_n):
        # flatten_width: [batch, channel, height, width]
        # batch_sample: [batch, camera, channel, height, width]
        flatten_width[:,:,:,width*i:width*(i+1)] = batch_sample[:,i,:,:,:]
    
    for i in range(batch_n):
        polar_image = warpper(flatten_width[i,:], reshape_size)
        polar_image = np.rot90(np.rot90(np.rot90(np.fliplr(polar_image))))
        output_tensor[i,:] = torch.Tensor(polar_image / 255).unsqueeze(0).permute(0,3,1,2) 
        
    return output_tensor

def backbone_pretrain(device, batch_sample, is_camera = True):

    pretrained_model = torchvision.models.resnet18(pretrained = True)
    modules = list(pretrained_model.children())[:-3]
    res_model = nn.Sequential(*modules)
    res_model.eval()

    if is_camera:
        # 5D tensor [batch, camera, C, H, W]
        camera_after_bb_list = []
        for i in range(batch_sample.shape[0]):
            camera_after_bb_ = res_model(batch_sample[i,:])
            # torch.Size([6, 256, 16, 20])
            print(camera_after_bb_.shape)
            camera_after_bb_list.append(camera_after_bb_)
            
        camera_after_bb = torch.stack(camera_after_bb_list, dim = 0)
        return camera_after_bb

    else:
        polar_after_bb = res_model(batch_sample)
        return polar_after_bb

    



if __name__ == "__main__":
    
    # [batch_size, 6(images per sample), 3, H, W]
    batch_sample = torch.rand((2, 6, 3, 256, 306))

    camera_out = camera_preprocess(batch_sample)
    # (8, 6, 3, 256, 306)
    print(camera_out.shape)
    
    polar_out = polar_preprocess(batch_sample)
    # [8, 3, 3000, 3000]
    print(polar_out.shape)

    after_bb_tensor = backbone_pretrain(device, batch_sample, is_camera = True)

    # camera: torch.Size([2, 6, 256, 16, 20])
    # polar: torch.Size([8, 256, 188, 188])
    print(after_bb_tensor.shape)
    
