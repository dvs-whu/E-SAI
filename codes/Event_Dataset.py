from torch.utils.data import Dataset
import torch
import numpy as np
import os 
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_file_name(path,suffix):
    ## function used to get file names
    name_list=[]
    file_list = os.listdir(path)
    for i in file_list:
        if os.path.splitext(i)[1] == suffix:
            name_list.append(i)
    name_list.sort()
    return name_list

class TestSet_Hybrid(Dataset): 
    def __init__(self, path_dir): 
        ## path_dir: directory path of test data
        self.path_dir = path_dir 
        self.event_names = get_file_name(self.path_dir,'.npy')
        self.event_names.sort()
    
    def __len__(self):
        return len(self.event_names)
    
    def __getitem__(self, index):
        
        ## load test data
        event_name = self.event_names[index]
        event_path = os.path.join(self.path_dir, event_name)
        event_data = np.load(event_path, allow_pickle=True).item()
        
        ## fuse Positive and Negative events
        pos = event_data['Pos']
        neg = event_data['Neg']
        pos = torch.FloatTensor(np.expand_dims(pos,axis=1))
        neg = torch.FloatTensor(np.expand_dims(neg,axis=1))
        event_input = torch.cat((pos,neg), 1)
                
        return event_input

class TestSet_AutoRefocus(Dataset):
    def __init__(self, PathDir): 
        self.PathDir = PathDir
        self.data_names = get_file_name(self.PathDir,'.npy') 
        self.data_names.sort()
    
    def __len__(self):
        return len(self.data_names)
    
    def __getitem__(self,index):
        
        event_name = self.data_names[index]
        
        ## warping event data
        event_path = os.path.join(self.PathDir, event_name)
        event_data = np.load(event_path, allow_pickle=True).item()
        pos = event_data['Pos']
        neg = event_data['Neg']
        diff_t = torch.FloatTensor(np.array(event_data['index_t'] - event_data['ref_t']))
        fx = event_data['fx']
        depth = event_data['depth']
        
        pos = np.expand_dims(pos,axis=1)
        neg = np.expand_dims(neg,axis=1)
        event_input = torch.FloatTensor(np.concatenate((pos, neg), axis=1)) ## event_input = (step, channel, H, W)
        
        return event_input, diff_t, depth, fx
        
