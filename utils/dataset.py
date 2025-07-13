import torch 
import pandas as pd 
import numpy as np 
from torch.utils.data import Dataset


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
        image = torch.tensor(image, dtype=torch.float32) / 255.0

        if self.transform:
            image = self.transform(image)
        
        return label,image