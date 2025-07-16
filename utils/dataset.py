import torch 
import pandas as pd 
import numpy as np 
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy


class FER2013Dataset(Dataset):

    def __init__(self,csv_path,usage='Training',transform=None):
        self.data = pd.read_csv(csv_path)
        self.data = self.data[self.data['Usage'] == usage]
        self.transform = transform 
    
    def __len__(self):
        return len(self.data)
    

    def __getitem__(self,idx):
        #load label 
        label = int(self.data.iloc[idx]['emotion'])
        pixel_seq = self.data.iloc[idx,2]
        image = np.fromstring(pixel_seq, sep=' ', dtype=np.uint8).reshape(48, 48)
        image = np.expand_dims(image,axis=0) #shape(1,48,48)

        #convert to tensor 
        image = torch.tensor(image, dtype=torch.float32) #should we divide by 255? TODO to think about it

        if self.transform:
            image = self.transform(image)
        
        return label,image

class DataModule():
    """
    Data Module class that wraps around Dataset and returns train,test and val loaders
    """
    def __init__(self,path, transform, **loader_kwargs):
        self.train_dataset = FER2013Dataset(path, usage='Training',transform=transform)
        #self.transform = transform
        self.test_dataset = FER2013Dataset(path,usage='PrivateTest')
        self.val_dataset = FER2013Dataset(path,usage='PublicTest')
        self.loader_kwargs = loader_kwargs
    def get_train_loader(self):
        return DataLoader(dataset=self.train_dataset,shuffle=True)
    
    def get_test_loader(self):
        return DataLoader(dataset=self.test_dataset, **self.loader_kwargs)
    
    def get_val_loader(self):
        return DataLoader(dataset=self.val_dataset, **self.loader_kwargs)