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
                    
        # (590436, 4, 2) = ((800 - 45 + 1) * (800 - 20 + 1), 4, 2)
        # print(out_array.shape)           
        out_array = np.array(out)
        self.bbox_gen = out_array
    
    def sample(self, k, input_bbox_array):
        total_n = self.bbox_gen.shape[0]
        pick_idx = random.sample(range(total_n), k)   
        
        # get positive coordinates
        points_pos = np.argwhere(input_bbox_array).tolist()
        
        # loop over each picked indices
        # if any cornor is contained in the positive coordinates, remove from the picked indices
        """
        for i in pick_idx:
            picked = self.bbox_gen[i, :]
            for coor in picked:
                if coor.tolist() in points_pos:
                    pick_idx.remove(i)
                    break
                    
        # continue to sample the remaining
        # sample logic but sample fewer and fewer if not having k sample bbox
        # may sample the same indices, check length with set()
        while len(set(pick_idx)) != k:
            pick_idx_new = random.sample(range(total_n), k - len(set(pick_idx)))
            for i in pick_idx_new:
                picked = self.bbox_gen[i, :]
                for coor in picked:
                    if coor.tolist() in points_pos:
                        pick_idx_new.remove(i)
                        break
            pick_idx.extend(pick_idx_new)
        """
        # may have more indices than k stored in pick_idx
        # remove duplicates to get exactly k indices
        pick_bbox = self.bbox_gen[list(set(pick_idx)), :]
        return pick_bbox
    
    def get_bbox(self):
        return self.bbox_gen
    


if __name__ == "__main__":
    # 800 * 800
    input_box = np.load('./obj_binary_roadmap/road_map/scene_106_sample_0.npy')
    
    roadmap_dim = (800,800)
    car_height = 20
    car_width = 45

    bbox_gen = BboxGenerate(roadmap_dim[0], roadmap_dim[1], car_height, car_width)
    bbox_all = bbox_gen.get_bbox()

    # (590436, 4, 2)
    print(bbox_all.shape)

    # len = 100
    print(len(bbox_gen.sample(100, input_box)))
