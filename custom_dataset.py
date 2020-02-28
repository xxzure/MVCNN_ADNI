from torch.utils.data.dataset import Dataset
import os
from PIL import Image
import pandas as pd

class MultiViewDataSet(Dataset):

    def find_target(self, dir):
        df=pd.read_csv('ADNI.csv')
        data_dict = {}      
        def map_dict(item):
            data_dict[str(item["ID"])] = [item["target"]]
        df.apply(map_dict,axis=1)
        
        return data_dict

    def __init__(self, root, data_type, transform=None, target_transform=None):
        self.x = []
        self.target = []
        # self.info = []
        self.root = root

        self.data_dict = self.find_target(root)

        self.transform = transform
        self.target_transform = target_transform
        # root / <train/test> / <item> / <view>.png
        for item in os.listdir(root + '/' + data_type):
            views = []
            path = root + '/' + data_type + '/' + item
            if os.path.isdir(path):
                for view in os.listdir(path):
                    views.append(path + '/' + view)
                self.x.append(views)
                self.target.append(self.data_dict[item][0])
                # self.info.append([self.data_dict[item][3],self.data_dict[item][4],self.data_dict[item][5],self.data_dict[item][6]])

    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):
        orginal_views = self.x[index]
        views = []

        for view in orginal_views:
            im = Image.open(view)
            im = im.convert('L')
            if self.transform is not None:
                im = self.transform(im)
            views.append(im)

        return views, self.target[index] #, self.info[index]

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.x)
