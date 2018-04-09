from torch.utils.data import Dataset
from scipy.io import loadmat
from PIL import Image
import numpy as np

class SunAttributeDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, mat_file, mat_attributes, root_dir, transform=None):
        """
        Args:
            mat_file (string): Path to the mat file with annotations.
            mat_attributes (string): Path to the mat file with attributes annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        img_paths = loadmat(mat_file)['images']
        length = img_paths.shape[0]
        class_num = 0
        label_dict = {}
        for i in range(length):
            path = img_paths[i][0][0]
            l = path.split('/')[1]
            if l not in label_dict.keys():
                label_dict[l] = class_num
                class_num += 1
                
        labels = np.zeros((length))
        for i in range(length):
            path = img_paths[i][0][0]
            l = path.split('/')[1]
            labels[i] = label_dict[l]
        
        self.label_dict = label_dict
        self.img_path = img_paths
        self.labels =  labels     
        self.img_attr = loadmat(mat_attributes)['labels_cv']
        self.length = length
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img = Image.open(self.root_dir + self.img_path[index][0][0])
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        
        label = self.labels[index]
        attri = self.img_attr[index]
        return img, label, attri
