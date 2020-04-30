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
                if j + self.car_height > 800 or i + self.car_width > 800:
                    break
                else:
                    box = np.array([left_top, right_top, left_bottom, right_bottom])
                    out.append(box)
        out_array = np.array(out)
        # (590436, 4, 2) = ((800 - 45 + 1) * (800 - 20 + 1), 4, 2)
        # print(out_array.shape)
        self.bbox_gen = out_array
    
    def sample_bbox(self, k, bbox_array):
        total_n = bbox_array.shape[0]
        pick_idx = random.sample(range(total_n), k)        
        pick_bbox = bbox_array[pick_idx, :]
        return pick_bbox
    
    def return_gen_bbox(self):
        return self.bbox_gen
    



if __name__ == "__main__":
    roadmap_dim = (800,800)
    car_height = 20
    car_width = 45

    bbox_gen = BboxGenerate(roadmap_dim[0], roadmap_dim[1], car_height, car_width)
    bbox_all = bbox_gen.return_gen_bbox()
    # (590436, 4, 2)
    print(bbox_all.shape)
    # len = 10
    print(len(bbox_gen.sample_bbox(10, bbox_all)))