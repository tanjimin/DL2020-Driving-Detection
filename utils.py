import numpy as np
import random
import torch
from scipy import misc, ndimage

class BboxGenerate():
    
    def __init__(self, roadmap_height, roadmap_width, car_height, car_width):
        self.roadmap_height = roadmap_height
        self.roadmap_width = roadmap_width
        self.car_height = car_height
        self.car_width = car_width
        
        out = []
        for i in range(self.roadmap_width):
            for j in range(self.roadmap_height):
                left_top = [i, j]
                left_bottom = [i, j + self.car_height]
                right_top = [i + self.car_width, j]
                right_bottom = [i + self.car_width, j + self.car_height]
                if j + self.car_height >= self.roadmap_height or i + self.car_width >= self.roadmap_width:
                    break
                else:
                    box = np.array([left_top, right_top, left_bottom, right_bottom])
                    out.append(box)
                    
        # (590436, 4, 2) = ((800 - 45 + 1) * (800 - 20 + 1), 4, 2)
        # print(out_array.shape)           
        out_array = np.array(out)
        self.bbox_gen = out_array
    
    def sample(self, k, input_bbox_array):
        total_n = self.bbox_gen.shape[0]
        pick_idx = random.sample(range(total_n), 2*k)   
        
        # picked: (2*k, 4, 2)
        picked = np.array(self.bbox_gen[pick_idx,:])
        # list of x coordinates, y coordinates from picked
        x_ = picked[:,:,0]
        y_ = picked[:,:,1]

        # create mask for picked
        # mask: (2*k,)
        mask = np.sum(input_bbox_array[x_, y_], axis = 1) == 0
        picked_masked = picked[mask]

        # (k, 4, 2)
        return picked_masked[:k,]
    
    def get_bbox(self):
        return self.bbox_gen
    

def concat_cameras(outputs):
    '''
    outputs: batch * 6 * 400 * 538
    camera orders are changed

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

def reorder_tensor(tensor, device):
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
    out_tensor = torch.zeros(tensor.shape).to(device)
    
    out_tensor[:,0,:] = tensor[:,0,:]
    out_tensor[:,1,:] = tensor[:,1,:]
    out_tensor[:,2,:] = tensor[:,2,:]
    out_tensor[:,3,:] = tensor[:,5,:]
    out_tensor[:,4,:] = tensor[:,4,:]
    out_tensor[:,5,:] = tensor[:,3,:]
    
    return out_tensor

if __name__ == "__main__":
    # 800 * 800
    input_box = np.load('/Users/leo/Downloads/DL_data/obj_binary_roadmap/road_map/scene_106_sample_0.npy')
    
    roadmap_dim = (800,800)
    car_height = 20
    car_width = 45

    bbox_gen = BboxGenerate(roadmap_dim[0], roadmap_dim[1], car_height, car_width)
    bbox_all = bbox_gen.get_bbox()

    # (590436, 4, 2)
    print(bbox_all.shape)

    # len = 100
    print(len(bbox_gen.sample(100, input_box)))
