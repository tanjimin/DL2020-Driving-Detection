import numpy as np
import random
import torch
from scipy import misc, ndimage

def reorder_tensor(tensor):
    '''
    Input/Output tensor shape: [batch_size * 6(images per sample), 3, H, W]
    
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
    for i in range(tensor.shape[0] // 6):
        out_tensor[0+i*6,:] = tensor[0+i*6,:]
        out_tensor[1+i*6,:] = tensor[1+i*6,:]
        out_tensor[2+i*6,:] = tensor[2+i*6,:]
        out_tensor[3+i*6,:] = tensor[5+i*6,:]
        out_tensor[4+i*6,:] = tensor[4+i*6,:]
        out_tensor[5+i*6,:] = tensor[3+i*6,:]
    
    return out_tensor

def concat_cameras(outputs):
    '''
    outputs: batch * 6 * 400 * 538
    camera orders are changed_order
    init_out: batch * 800 * 800
    '''
    batch_n = outputs.shape[0]

    front_left = outputs[:,0,:]
    front = outputs[:,1,:]
    front_right = outputs[:,2,:]
    back_right = outputs[:,3,:]
    back = outputs[:,4,:]
    back_left = outputs[:,5,:]

    rot_axes = (2,1)

    init_out = np.zeros((batch_n, 800,800))
    # rotate counter-clockwise
    # assign front left and back right cameras
    # then rotate clockwise back and crop out 800*800
    init_out = ndimage.rotate(init_out, 30, axes = rot_axes)
    init_out[:, 547 - 400:547, 278: 278 + 538] += front_left.copy()
    init_out[:, 547:547+400, 278: 278 + 538] += back_right[:, ::-1,::-1].copy()
    init_out = ndimage.rotate(init_out, -30, axes = rot_axes)
    init_out = init_out[:, 347:347+800, 347:347+800]

    # rotate clockwise
    # assign front right and back right cameras
    # then rotate counter-clockwise back and crop out 800*800
    init_out = ndimage.rotate(init_out, -30, axes = rot_axes)
    init_out[:, 547-400:547, 278: 278 + 538 ] += back_left.copy()
    init_out[:, 547:547+400, 278: 278 + 538 ] += front_right[:, ::-1,::-1].copy()
    init_out = ndimage.rotate(init_out, 30, axes = rot_axes)
    init_out = init_out[:, 347:347+800, 347:347+800]

    # inverse the process for front/back as well
    init_out[:, 131:131+538, 400:] += ndimage.rotate(front, -90, axes = rot_axes)
    init_out[:, 131:131+538, :400] += ndimage.rotate(back[:, ::-1,::-1], -90, axes = rot_axes)

    init_out = torch.FloatTensor(init_out)

    return init_out