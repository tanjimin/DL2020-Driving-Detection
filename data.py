import torch
import torchvision
import torchvision.transforms as transforms

class LaneSegmentationDataset(Dataset):
    
    def __init__(self, data_dir, label_dir):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.data_names = sorted(os.listdir(data_dir))

    def __len__(self):
        return len(self.file_names)

    def __get_item__(self, idx):
        data_path = os.path.join(data_dir, data_names[idx])
        label_path = os.path.join(label_dir, data_names[idx])
        data = np.read(data_path)
        label = np.read(label_path)
        return data, label
        
