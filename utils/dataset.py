import torch 
import pandas as pd 
import numpy as np 
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy


class FER2013Dataset(Dataset):

    def __init__(self,csv_path,model_name,usage='Training',transform=None):

        self.data = pd.read_csv(csv_path)
        self.data = self.data[self.data['Usage'] == usage]
        self.transform = transform 
        self.model_name = model_name
        
    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, idx):
        label = int(self.data.iloc[idx]['emotion'])
        pixel_seq = self.data.iloc[idx, 2]
        image = np.fromstring(pixel_seq, sep=' ', dtype=np.uint8).reshape(48, 48)

        # Convert grayscale (48,48) to RGB (48,48,3)
        if self.model_name == "resnet_fer":
            image = np.stack([image]*3, axis=-1)  # (48, 48, 3)
        
            if self.transform:
                image = self.transform(image)  # transform expects HWC or PIL image
                
            else:
                # convert to tensor channel-first if no transform
                image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0  # (3,48,48)
        else: 
            image = np.expand_dims(image,axis=0)
            image = torch.tensor(image, dtype=torch.float32) / 255.0
            if self.transform:
                image = self.transform(image) 

        return label, image


class DataModule():
    """
    Data Module class that wraps around Dataset and returns train,test and val loaders
    """
    def __init__(self,path, model_name, transform, **loader_kwargs):
        
        self.train_dataset = FER2013Dataset(path,model_name, usage='Training',transform=transform)
        #self.transform = transform
        self.test_dataset = FER2013Dataset(path,model_name,usage='PrivateTest')
        self.val_dataset = FER2013Dataset(path,model_name,usage='PublicTest')
        self.loader_kwargs = loader_kwargs
    def get_train_loader(self):
        return DataLoader(dataset=self.train_dataset,shuffle=True, batch_size=self.loader_kwargs.get('batch_size', 2048))
    
    def get_test_loader(self):
        return DataLoader(dataset=self.test_dataset, **self.loader_kwargs)
    
    def get_val_loader(self):
        return DataLoader(dataset=self.val_dataset, **self.loader_kwargs)