import numpy as np
import random

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
                if j + self.car_height >= 800 or i + self.car_width >= 800:
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
